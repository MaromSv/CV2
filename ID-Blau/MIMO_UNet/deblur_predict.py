import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import sys
import tqdm
import argparse
import matplotlib.pyplot as plt
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from dataloader import Test_Loader
from MIMOUNet import build_MIMOUnet_net
from utils.utils import same_seed, count_parameters, judge_and_remove_module_dict


@torch.no_grad()
def predict(model, args, device):
    model.eval()

    if args.dataset == 'GoPro+HIDE':
        dataset_name = ['GoPro', 'HIDE']
    else:
        dataset_name = [args.dataset]

    for val_dataset_name in dataset_name:
        dataset_path = os.path.join(args.data_path, val_dataset_name)

        dataset = Test_Loader(data_path=dataset_path,
                              crop_size=args.crop_size,
                              ZeroToOne=False)
        
        # Create output directories
        save_dir = os.path.join(args.dir_path, 'results', f'{val_dataset_name}')
        os.makedirs(os.path.join(save_dir, 'mag'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'dx'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'dy'), exist_ok=True)
        
        dataset_len = len(dataset)
        tq = tqdm.tqdm(range(dataset_len))
        tq.set_description(f'Predict {val_dataset_name}')

        for idx in tq:
            sample = dataset[idx]
            input = sample['blur'].unsqueeze(0).to(device)
            b, c, h, w = input.shape
            factor = 8
            h_n = (factor - h % factor) % factor
            w_n = (factor - w % factor) % factor
            input = torch.nn.functional.pad(input, (0, w_n, 0, h_n), mode='reflect')

            # Get model outputs (tuples of dx, dy, mag at each scale)
            outputs = model(input)
            
            # Use full resolution output (last in the list)
            dx, dy, mag = outputs[2]
            
            # Crop to original size
            dx = dx[:, :, :h, :w].clamp(-0.5, 0.5)
            dy = dy[:, :, :h, :w].clamp(-0.5, 0.5)
            mag = mag[:, :, :h, :w].clamp(-0.5, 0.5)
            
            # Get image name
            image_name = os.path.split(dataset.get_path(idx=idx)['blur_path'])[-1]
            base_name = os.path.splitext(image_name)[0]
            
            # Save magnitude
            save_mag_path = os.path.join(save_dir, 'mag', f"{base_name}.png")
            save_image(mag + 0.5, save_mag_path)
            
            # Save dx (displacement in x direction)
            save_dx_path = os.path.join(save_dir, 'dx', f"{base_name}.png")
            save_image(dx + 0.5, save_dx_path)
            
            # Save dy (displacement in y direction)
            save_dy_path = os.path.join(save_dir, 'dy', f"{base_name}.png")
            save_image(dy + 0.5, save_dy_path)



if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default='./dataset/test', type=str)
    parser.add_argument("--dir_path", default='./out/MIMO-UNetPlus', type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--model", default='MIMO-UNetPlus', type=str, choices=['MIMO-UNet', 'MIMO-UNetPlus'])
    parser.add_argument("--dataset", default='GoPro+HIDE', type=str, choices=['GoPro+HIDE', 'GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R', 'RWBI'])
    parser.add_argument("--crop_size", default=None, type=int)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    load_model_state = torch.load(args.model_path)

    if not os.path.isdir(args.dir_path):
        os.makedirs(args.dir_path)

    # Model and optimizer
    net = build_MIMOUnet_net(args.model)
    
    
    load_model_state = torch.load(args.model_path)

    if 'model_state' in load_model_state.keys():
        load_model_state["model_state"] = judge_and_remove_module_dict(load_model_state["model_state"])
        net.load_state_dict(load_model_state["model_state"])
    elif 'model' in load_model_state.keys():
        load_model_state["model"] = judge_and_remove_module_dict(load_model_state["model"])
        net.load_state_dict(load_model_state["model"])
    else:
        load_model_state = judge_and_remove_module_dict(load_model_state)
        net.load_state_dict(load_model_state)

    net = nn.DataParallel(net)
    net.to(device)

    print("device:", device)
    print(f'args: {args}')
    print(f'model: {net}')
    print(f'model parameters: {count_parameters(net)}')

    same_seed(2023)
    predict(net, args=args, device=device)





