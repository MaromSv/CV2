import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sys
import tqdm
import cv2
import os
import argparse
import logging
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from dataloader import Multi_GoPro_Loader
from MIMOUNet import build_MIMOUnet_net
from models.losses import CharbonnierLoss, VGGPerceptualLoss, L1andPerceptualLoss
from utils.utils import calc_psnr, same_seed, count_parameters, tensor2cv, AverageMeter, judge_and_remove_module_dict
import torch.nn.functional as F
import pyiqa
from tensorboardX import SummaryWriter

cv2.setNumThreads(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def rfft(x, dim):
    """Real FFT for PyTorch tensors"""
    return torch.fft.rfftn(x, dim=dim)

class Trainer():
    def __init__(self, dataloader_train, dataloader_val, model, optimizer, scheduler, args, writer) -> None:
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.writer = writer
        self.epoch = 0
        self.device = self.args.device
        self.psnr_func = pyiqa.create_metric('psnr', device=device)
        self.lpips_func = pyiqa.create_metric('lpips', device=device)
        self.best_psnr = args.best_psnr if hasattr(args, 'best_psnr') else 0
        self.grad_clip = 1

        self.scheduler.T_max = self.args.end_epoch
        if args.criterion == "l1":
            self.criterion = CharbonnierLoss()
        elif args.criterion == "perceptual":
            self.criterion = VGGPerceptualLoss().to(device)
        elif args.criterion == "l1perceptual":
             self.criterion = L1andPerceptualLoss(gamma=args.gamma).to(device)
        else:
            raise ValueError(f"criterion not supported {args.criterion}")
        
    def train(self):
        if dist.get_rank() == 0:
            print('###########################################')
            print('Start_Epoch:', self.args.start_epoch)
            print('End_Epoch:', self.args.end_epoch)
            print('Model:', self.args.model_name)
            print(f"Optimizer:{self.optimizer.__class__.__name__}")
            print(f"Scheduler:{self.scheduler.__class__.__name__ if self.scheduler else None}")
            print(f"Train Data length:{len(dataloader_train.dataset)}")
            print("start train !!")
            print('###########################################')

        for epoch in range(args.start_epoch, args.end_epoch + 1):
            self.epoch = epoch
            self._train_epoch()

            if dist.get_rank() == 0:
                if (epoch % self.args.validation_epoch) == 0 or epoch == self.args.end_epoch:
                    self.valid()

                if(self.args.val_save_epochs > 0 and epoch % self.args.val_save_epochs == 0 or epoch == self.args.end_epoch):
                    self.val_save_image(dir_path=self.args.dir_path, dataset=self.dataloader_val.dataset)

                self.save_model()
    
    def _train_epoch(self):
        train_sampler.set_epoch(self.epoch)
        tq = tqdm.tqdm(self.dataloader_train, total=len(self.dataloader_train))
        tq.set_description(f'Epoch [{self.epoch}/{self.args.end_epoch}] training')
        total_train_loss = AverageMeter()
        total_train_psnr = AverageMeter()
        
        for idx, sample in enumerate(tq):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Get blur image and ground truth blur field (only at highest resolution)
            blur, gt_field = sample['blur'].to(self.device), sample['blur_field'].to(self.device)
            outs = self.model(blur)  # list of (dx,dy,mag) tuples
            
            # Get only the highest resolution prediction (last in the list)
            dx, dy, mag = outs[2]
            pred = torch.cat([dx, dy, mag], dim=1)
            
            # Compute loss only at the highest resolution
            loss_content = self.criterion(pred, gt_field)
            
            # FFT loss (optional)
            loss_fft = 0
            if self.args.use_fft_loss:
                pred_fft = rfft(pred, 2)
                gt_fft = rfft(gt_field, 2)
                loss_fft = self.criterion(pred_fft, gt_fft)
                loss = loss_content + 0.1 * loss_fft
            else:
                loss = loss_content
            
            loss.backward()
            self.optimizer.step()
            
            total_train_loss.update(loss.detach().item())
            
            # Calculate PSNR on magnitude channel only for visualization
            mag_psnr = calc_psnr(pred[:, 2:3, :, :].detach(), gt_field[:, 2:3, :, :].detach())
            total_train_psnr.update(mag_psnr)
            
            tq.set_postfix({'loss': total_train_loss.avg, 'mag_psnr': total_train_psnr.avg, 'lr': self.optimizer.param_groups[0]['lr']})

        if self.scheduler:
            self.scheduler.step()
        if self.writer and dist.get_rank() == 0:
            self.writer.add_scalar('Loss/Train_loss', total_train_loss.avg, self.epoch)
            self.writer.add_scalar('Loss/Train_mag_psnr', total_train_psnr.avg, self.epoch)
            logging.info(
                f'Epoch [{self.epoch}/{self.args.end_epoch}]: Train_loss: {total_train_loss.avg:.4f} Train_mag_psnr:{total_train_psnr.avg:.4f}')
    
    @torch.no_grad()
    def _valid(self, blur, gt_field):
        self.model.eval()
        outs = self.model(blur)
        
        # Get only the highest resolution prediction
        dx, dy, mag = outs[2]
        pred = torch.cat([dx, dy, mag], dim=1)
        
        # Compute loss only at the highest resolution
        loss_content = self.criterion(pred, gt_field)
        
        # FFT loss (optional)
        if self.args.use_fft_loss:
            pred_fft = rfft(pred, 2)
            gt_fft = rfft(gt_field, 2)
            loss_fft = self.criterion(pred_fft, gt_fft)
            loss = loss_content + 0.1 * loss_fft
        else:
            loss = loss_content
        
        # Calculate PSNR on magnitude channel only
        mag_psnr = torch.mean(self.psnr_func(pred[:, 2:3, :, :].detach(), gt_field[:, 2:3, :, :].detach())).item()
        
        # For LPIPS, we'll just use the magnitude channel
        mag_lpips = torch.mean(self.lpips_func(pred[:, 2:3, :, :].detach(), gt_field[:, 2:3, :, :].detach())).item()
        
        return mag_psnr, mag_lpips, loss.item()
    
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        total_val_psnr = AverageMeter()
        total_val_lpips = AverageMeter()
        total_val_loss = AverageMeter()
        tq = tqdm.tqdm(self.dataloader_val, total=len(self.dataloader_val))
        tq.set_description(f'Epoch [{self.epoch}/{self.args.end_epoch}] Validation')
        for idx, sample in enumerate(tq):
            blur, gt_field = sample['blur'].to(self.device), sample['blur_field'].to(self.device)
            psnr, lpips, loss = self._valid(blur, gt_field)
            total_val_psnr.update(psnr)
            total_val_lpips.update(lpips)
            total_val_loss.update(loss)
            tq.set_postfix(LPIPS=total_val_lpips.avg, PSNR=total_val_psnr.avg, Loss=total_val_loss.avg)

        self.writer.add_scalar('Val/Test_lpips', total_val_lpips.avg, self.epoch)
        self.writer.add_scalar('Val/Test_psnr', total_val_psnr.avg, self.epoch)
        self.writer.add_scalar('Val/Test_loss', total_val_loss.avg, self.epoch)
        logging.info(
            f'Validation Epoch [{self.epoch}/{self.args.end_epoch}]: Test Loss: {total_val_loss.avg:.4f} Test lpips: {total_val_lpips.avg:.4f} Test psnr:{total_val_psnr.avg:.4f}')
        
        if self.best_psnr < total_val_psnr.avg:
            self.best_psnr = total_val_psnr.avg
            self.args.best_psnr = self.best_psnr
            best_state = {'model_state': self.model.module.state_dict(), 'args': self.args}
            torch.save(best_state, os.path.join(self.args.dir_path, f'best_{self.args.model_name}.pth'))

            print(f'Saving model with best PSNR {self.best_psnr:.3f}...')
            logging.info(f'Saving model with best PSNR {self.best_psnr:.3f}...')
            
    def save_model(self):
        """save model parameters"""
        training_state = {'epoch': self.epoch, 
                          'model_state': self.model.module.state_dict(),
                          'optimizer_state': self.optimizer.state_dict(),
                          'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                          'best_panr': self.best_psnr,
                          'args': args}
        torch.save(training_state, os.path.join(self.args.dir_path, 'last_{}.pth'.format(self.args.model_name)))

        if (self.epoch % self.args.check_point_epoch) == 0:
            torch.save(training_state, os.path.join(self.args.dir_path, 'epoch_{}_{}.pth'.format(self.epoch, self.args.model_name)))

        if self.epoch == self.args.end_epoch:
            model_state = {'model_state': self.model.module.state_dict(), 'args': args}
            torch.save(model_state, os.path.join(args.dir_path, 'final_{}.pth'.format(args.model_name)))

    @torch.no_grad()
    def val_save_image(self, dir_path, dataset, val_num=3):
        """Save visualization of blur field predictions"""
        os.makedirs(dir_path, exist_ok=True)
        self.model.eval()
        for idx in random.sample(range(0, len(dataset)), val_num):
            sample = dataset[idx]
            blur = sample['blur'].unsqueeze(0).to(self.device)
            gt_field = sample['blur_field'].unsqueeze(0).to(self.device)
            
            b, c, h, w = blur.shape
            factor = 8
            h_n = (factor - h % factor) % factor
            w_n = (factor - w % factor) % factor
            blur = torch.nn.functional.pad(blur, (0, w_n, 0, h_n), mode='reflect')
            
            outputs = self.model(blur)  # [(dx,dy,mag), (dx,dy,mag), (dx,dy,mag)]
            dx, dy, mag = outputs[2]  # Use full resolution output
            
            # Crop to original size
            dx = dx[:, :, :h, :w]
            dy = dy[:, :, :h, :w]
            mag = mag[:, :, :h, :w]
            
            # Clamp magnitude
            mag = mag.clamp(-0.5, 0.5)
            
            # Create visualization directories
            save_dir_path = os.path.join(dir_path, f'visualization')
            os.makedirs(os.path.join(save_dir_path, 'mag'), exist_ok=True)
            os.makedirs(os.path.join(save_dir_path, 'dx'), exist_ok=True)
            os.makedirs(os.path.join(save_dir_path, 'dy'), exist_ok=True)
            os.makedirs(os.path.join(save_dir_path, 'gt_mag'), exist_ok=True)
            
            # Save magnitude
            save_mag_path = os.path.join(save_dir_path, 'mag', f'{self.epoch:05d}_{idx:05d}.png')
            mag_img = tensor2cv(mag + 0.5)  # Convert to 0-1 range
            cv2.imwrite(save_mag_path, mag_img)
            
            # Save dx
            save_dx_path = os.path.join(save_dir_path, 'dx', f'{self.epoch:05d}_{idx:05d}.png')
            dx_img = tensor2cv(dx + 0.5)  # Convert to 0-1 range
            cv2.imwrite(save_dx_path, dx_img)
            
            # Save dy
            save_dy_path = os.path.join(save_dir_path, 'dy', f'{self.epoch:05d}_{idx:05d}.png')
            dy_img = tensor2cv(dy + 0.5)  # Convert to 0-1 range
            cv2.imwrite(save_dy_path, dy_img)
            
            # Save ground truth magnitude
            save_gt_mag_path = os.path.join(save_dir_path, 'gt_mag', f'{self.epoch:05d}_{idx:05d}.png')
            gt_mag = gt_field[0, 2:3, :, :]  # Extract magnitude channel
            gt_mag_img = tensor2cv(gt_mag + 0.5)  # Convert to 0-1 range
            cv2.imwrite(save_gt_mag_path, gt_mag_img)

if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_epoch", default=3000, type=int)
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--validation_epoch", default=25, type=int)
    parser.add_argument("--check_point_epoch", default=100, type=int)
    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--gamma", default=0.5, type=float)
    parser.add_argument("--optimizer", default='adam', type=str)
    parser.add_argument("--criterion", default='l1', type=str)
    parser.add_argument("--data_path", default='./dataset/GOPRO_Large', type= str)
    parser.add_argument("--generate_path", default=None, type=str, nargs='+')
    parser.add_argument("--dir_path", default='./experiments/MIMO_UNetPlus', type=str)
    parser.add_argument("--model_name", default='MIMO_UNetPlus', type=str)
    parser.add_argument("--model", default='MIMO-UNetPlus', type=str, choices=['MIMO-UNet', 'MIMO-UNetPlus'])
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--val_save_epochs", default=100, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--only_use_generate_data", action='store_true', help="only use generated data to train model.")
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--use_fft_loss", action='store_true', help="Use FFT loss in addition to spatial loss")
    
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')
    net = build_MIMOUnet_net(args.model)

    # training seed
    seed = args.seed + args.local_rank
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args.device = device
    print("device:", device)
    num_gpus = torch.cuda.device_count()
    net.to(device)

    print(args.__dict__.items())

    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.init_lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=args.init_lr, weight_decay=1e-4)
    else:
        raise ValueError(f"optimizer not supported {args.optimizer}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.end_epoch, eta_min=args.min_lr)
    # load pretrained
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    if os.path.exists(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name))):
        print('load_pretrained')
        training_state = (torch.load(os.path.join(args.dir_path, 'last_{}.pth'.format(args.model_name)), map_location=map_location))
        args.start_epoch = training_state['epoch'] + 1
        if 'best_psnr' in training_state['args']:
            args.best_psnr = training_state['args'].best_psnr
        new_weight = net.state_dict()
        training_state["model_state"] = judge_and_remove_module_dict(training_state["model_state"])
        new_weight.update(training_state['model_state'])
        net.load_state_dict(new_weight)
        new_optimizer = optimizer.state_dict()
        new_optimizer.update(training_state['optimizer_state'])
        optimizer.load_state_dict(new_optimizer)
        new_scheduler = scheduler.state_dict()
        new_scheduler.update(training_state['scheduler_state'])
        scheduler.load_state_dict(new_scheduler)
    elif args.resume:
        print('load_resume_pretrained')
        model_load = torch.load(args.resume, map_location=map_location)
        if 'model_state' in model_load.keys():
            model_load["model_state"] = judge_and_remove_module_dict(model_load["model_state"])
            net.load_state_dict(model_load['model_state'])
        else:
            model_load = judge_and_remove_module_dict(model_load)
            net.load_state_dict(model_load)
        os.makedirs(args.dir_path, exist_ok=True)
    else:
        os.makedirs(args.dir_path, exist_ok=True)
    
    # Model
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                          output_device=args.local_rank)
    
    # Traning loader
    train_data_path = args.data_path
    if args.only_use_generate_data:
        train_data_path = None
    Train_set = Multi_GoPro_Loader(data_path=train_data_path, generate_path=args.generate_path, mode="train", crop_size=args.crop_size)
    train_sampler = DistributedSampler(Train_set)
    dataloader_train = DataLoader(Train_set, sampler=train_sampler, batch_size=args.batch_size//num_gpus, num_workers=8, pin_memory=True)

    # Val loader
    Val_set = Multi_GoPro_Loader(data_path=args.data_path, generate_path=args.generate_path, mode="test", crop_size=args.crop_size)
    dataloader_val = DataLoader(Val_set, batch_size=args.batch_size//num_gpus, shuffle=True, num_workers=8,
                                drop_last=False, pin_memory=True)
    writer = None
    if dist.get_rank() == 0:

        logging.basicConfig(
            filename=os.path.join(args.dir_path, 'train.log') , format='%(levelname)s:%(message)s', encoding='utf-8', level=logging.INFO)
        
        logging.info(f'args: {args}')
        logging.info(f'model: {net}')
        logging.info(f'model parameters: {count_parameters(net)}')
        logging.info(f"Optimizer:{optimizer.__class__.__name__}")
        logging.info(f"Train Data length:{len(dataloader_train.dataset)}")

        writer = SummaryWriter(os.path.join("MIMO_log", args.model_name))
        writer.add_text("args", str(args))

    trainer = Trainer(dataloader_train, dataloader_val, net, optimizer, scheduler, args, writer)
    trainer.train()
    

    

