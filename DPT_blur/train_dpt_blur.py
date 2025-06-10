import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.transforms import Compose
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm
import random # For dataset filtering example
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Import DPT model and transforms
try:
    from dpt_lib.models import DPT
    from dpt_lib.transforms import Resize, NormalizeImage, PrepareForNet
    from dpt_lib.blocks import Interpolate
except ImportError as e:
    print("Error: Could not import local DPT library (dpt_lib). Make sure it exists in the same directory as the script.")

# --- Import Dataset Class ---
from data_loader import BlurMapDataset

# --- Import Model Creation Utility ---
from model_utils import create_dpt_blur_model

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        
    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()

class BlurVectorLoss(nn.Module):
    """
    Improved loss function for blur vectors combining:
    1. Charbonnier loss for magnitude prediction
    2. Sigmoid-transformed MSE for full vector field
    3. Multi-scale consistency loss (if using multi-scale predictions)
    """
    def __init__(self, use_direction=False, direction_weight=1.0, magnitude_weight=1.0, 
                 consistency_weight=0.1, eps=1e-6):
        super(BlurVectorLoss, self).__init__()
        self.use_direction = use_direction
        self.direction_weight = direction_weight if use_direction else 0.0
        self.magnitude_weight = magnitude_weight
        self.consistency_weight = consistency_weight
        self.eps = eps
        self.charbonnier = CharbonnierLoss(eps=eps)
        
    def forward(self, pred, target):
        # Handle multi-scale predictions
        if isinstance(pred, list):
            return self._forward_multi_scale(pred, target)
        return self._forward_single_scale(pred, target)
    
    def _forward_single_scale(self, pred, target):
        # Extract components
        pred_bx, pred_by, pred_mag = pred[:, 0], pred[:, 1], pred[:, 2]
        target_bx, target_by, target_mag = target[:, 0], target[:, 1], target[:, 2]
        
        # Ensure positive magnitude predictions
        pred_mag = torch.abs(pred_mag)
        
        # 1. Charbonnier loss for magnitude
        magnitude_loss = self.charbonnier(pred_mag, target_mag)
        
        # 2. Sigmoid-transformed MSE for full vector field
        pred_sigmoid = torch.sigmoid(pred)
        mse_loss = F.mse_loss(pred_sigmoid, target)
        
        # Calculate MSE metrics for evaluation
        mse_total = torch.mean((pred - target) ** 2)
        mse_magnitude = torch.mean((pred_mag - target_mag) ** 2)
        mse_direction = torch.mean((pred_bx - target_bx) ** 2 + (pred_by - target_by) ** 2)
        mse_bx = torch.mean((pred_bx - target_bx) ** 2)
        mse_by = torch.mean((pred_by - target_by) ** 2)
        
        # Initialize direction loss and related metrics
        direction_loss = torch.tensor(0.0, device=pred.device)
        cos_sim = torch.tensor(1.0, device=pred.device)
        angle_diff = torch.tensor(0.0, device=pred.device)
        
        if self.use_direction:
            # Calculate vector statistics
            pred_norm = torch.sqrt(pred_bx**2 + pred_by**2 + self.eps)
            target_norm = torch.sqrt(target_bx**2 + target_by**2 + self.eps)
            
            # Normalize vectors for direction comparison
            pred_bx_norm = pred_bx / pred_norm
            pred_by_norm = pred_by / pred_norm
            target_bx_norm = target_bx / target_norm
            target_by_norm = target_by / target_norm
            
            # Direction loss using cosine similarity
            cos_sim = (pred_bx_norm * target_bx_norm + pred_by_norm * target_by_norm).clamp(-1.0 + self.eps, 1.0 - self.eps)
            direction_loss = (1.0 - cos_sim).mean()
            angle_diff = torch.acos(cos_sim)  # Angle difference in radians
        
        # Calculate total loss
        total_loss = (self.magnitude_weight * magnitude_loss + 
                     0.5 * mse_loss +  # MSE component is weighted by 0.5 as per the paper
                     self.direction_weight * direction_loss)
        
        # Calculate additional statistics
        stats = {
            'magnitude_loss': magnitude_loss.item(),
            'mse_loss': mse_loss.item(),
            'total_loss': total_loss.item(),
            'pred_mag_mean': pred_mag.mean().item(),
            'pred_mag_std': pred_mag.std().item(),
            'pred_mag_max': pred_mag.max().item(),
            'target_mag_mean': target_mag.mean().item(),
            'target_mag_std': target_mag.std().item(),
            'target_mag_max': target_mag.max().item(),
            'mse_total': mse_total.item(),
            'mse_magnitude': mse_magnitude.item(),
            'mse_direction': mse_direction.item(),
            'mse_bx': mse_bx.item(),
            'mse_by': mse_by.item(),
            'psnr': (10 * torch.log10(1.0 / (mse_total + self.eps))).item()
        }
        
        if self.use_direction:
            stats.update({
                'direction_loss': direction_loss.item(),
                'cosine_sim_mean': cos_sim.mean().item(),
                'angle_diff_mean': angle_diff.mean().item()
            })
        
        return total_loss, stats
    
    def _forward_multi_scale(self, preds, target):
        """
        Handle multi-scale predictions with consistency loss
        preds: List of predictions at different scales
        target: Ground truth at the finest scale
        """
        total_loss = 0.0
        all_stats = []
        
        # Process each scale
        for i, pred in enumerate(preds):
            # Create downsampled target for this scale
            if pred.shape[-2:] != target.shape[-2:]:
                scaled_target = F.interpolate(target, size=pred.shape[-2:], 
                                           mode='bilinear', align_corners=False)
            else:
                scaled_target = target
            
            # Calculate loss for this scale
            scale_loss, scale_stats = self._forward_single_scale(pred, scaled_target)
            total_loss += scale_loss
            all_stats.append(scale_stats)
        
        # Add consistency loss between scales
        consistency_loss = 0.0
        for i in range(len(preds) - 1):
            # Upsample coarser prediction to match finer scale
            upsampled = F.interpolate(preds[i], size=preds[i+1].shape[-2:], 
                                    mode='bilinear', align_corners=False)
            # L1 loss between upsampled and finer prediction
            consistency_loss += torch.mean(torch.abs(upsampled - preds[i+1]))
        
        total_loss += self.consistency_weight * consistency_loss
        
        # Average statistics across scales
        avg_stats = {}
        for key in all_stats[0].keys():
            avg_stats[key] = sum(stat[key] for stat in all_stats) / len(all_stats)
        avg_stats['consistency_loss'] = consistency_loss.item()
        
        return total_loss, avg_stats

# --- Define collate_fn at the top level ---
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None # Return None if the whole batch is invalid
    return torch.utils.data.dataloader.default_collate(batch)

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, start_epoch, epochs, checkpoint_dir, best_val_loss):
    """Main training loop with checkpoint saving."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize statistics tracking
    all_stats = {
        'train': [],
        'val': []
    }

    # Check if we're using direction prediction
    use_direction = criterion.use_direction

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        print(f"\n=== Epoch {current_epoch}/{epochs} ===")

        # --- Training Phase ---
        model.train()
        epoch_stats = {
            'magnitude_loss': 0.0,
            'mse_loss': 0.0,
            'total_loss': 0.0,
            'pred_mag_mean': 0.0,
            'pred_mag_std': 0.0,
            'pred_mag_max': 0.0,
            'target_mag_mean': 0.0,
            'target_mag_std': 0.0,
            'target_mag_max': 0.0,
            'mse_total': 0.0,
            'mse_magnitude': 0.0,
            'mse_direction': 0.0,
            'mse_bx': 0.0,
            'mse_by': 0.0,
            'psnr': 0.0
        }
        
        # Add direction-related statistics only if using direction
        if use_direction:
            epoch_stats.update({
                'direction_loss': 0.0,
                'cosine_sim_mean': 0.0,
                'angle_diff_mean': 0.0
            })
        
        pbar_train = tqdm(train_loader, desc=f"Training", leave=False)
        batch_count = 0
        for batch_idx, batch in enumerate(pbar_train):
            if batch is None:
                print(f"Warning: Skipping None batch at index {batch_idx}")
                continue
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, list):
                # Handle multi-scale outputs
                loss, batch_stats = criterion(outputs, targets)
            else:
                # Handle single-scale output
                if outputs.shape[-2:] != targets.shape[-2:]:
                    outputs_resized = F.interpolate(outputs, size=targets.shape[-2:], 
                                                 mode='bilinear', align_corners=False)
                else:
                    outputs_resized = outputs
                loss, batch_stats = criterion(outputs_resized, targets)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered at Epoch {current_epoch}, Batch {batch_idx}. Skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()

            # Accumulate statistics
            batch_size = inputs.size(0)
            for key in epoch_stats:
                if key in batch_stats:  # Only update if key exists in batch_stats
                    epoch_stats[key] += batch_stats[key] * batch_size
            batch_count += batch_size

            # Update progress bar
            pbar_train.set_postfix({
                'loss': loss.item(),
                'mag_loss': batch_stats['magnitude_loss'],
                'mse': batch_stats['mse_total']
            })

        # Average statistics over the epoch
        for key in epoch_stats:
            epoch_stats[key] /= batch_count

        # Store training statistics
        all_stats['train'].append(epoch_stats)

        # --- Validation Phase ---
        model.eval()
        val_stats = {key: 0.0 for key in epoch_stats}
        val_batch_count = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Validation", leave=False)
            for batch_idx, batch in enumerate(pbar_val):
                if batch is None:
                    continue
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                
                if isinstance(outputs, list):
                    # Handle multi-scale outputs
                    loss, batch_stats = criterion(outputs, targets)
                else:
                    # Handle single-scale output
                    if outputs.shape[-2:] != targets.shape[-2:]:
                        outputs_resized = F.interpolate(outputs, size=targets.shape[-2:], 
                                                     mode='bilinear', align_corners=False)
                    else:
                        outputs_resized = outputs
                    loss, batch_stats = criterion(outputs_resized, targets)

                # Accumulate statistics
                batch_size = inputs.size(0)
                for key in val_stats:
                    if key in batch_stats:
                        val_stats[key] += batch_stats[key] * batch_size
                val_batch_count += batch_size

                # Update progress bar
                pbar_val.set_postfix({
                    'loss': loss.item(),
                    'mag_loss': batch_stats['magnitude_loss'],
                    'mse': batch_stats['mse_total']
                })

        # Average validation statistics
        for key in val_stats:
            val_stats[key] /= val_batch_count

        # Store validation statistics
        all_stats['val'].append(val_stats)

        # Print epoch statistics
        print(f"\nEpoch {current_epoch} Statistics:")
        print("Training:")
        for key, value in epoch_stats.items():
            print(f"  {key}: {value:.6f}")
        print("\nValidation:")
        for key, value in val_stats.items():
            print(f"  {key}: {value:.6f}")

        # Save checkpoint if validation loss improved
        if val_stats['total_loss'] < best_val_loss:
            best_val_loss = val_stats['total_loss']
            checkpoint = {
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'stats': all_stats
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"\nSaved new best model with validation loss: {best_val_loss:.6f}")

        # Save regular checkpoint
        checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'stats': all_stats
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{current_epoch}.pth'))

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats['total_loss'])
            else:
                scheduler.step()

    return best_val_loss, all_stats

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DPT model for blur map prediction.")
    # Data Args
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Base directory of the restructured dataset (e.g., data/dataset_DPT_blur/), \
                              which should contain train/blur, train/condition, val/blur, val/condition subdirectories.')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of training data to use for validation.')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for splitting training data into train/val.')
    # Model Args
    parser.add_argument('--weights', type=str, default='weights/dpt_hybrid-ade20k-53898607.pt', help='Path to pre-trained DPT segmentation weights (.pt file) for backbone initialization.')
    parser.add_argument('--model_type', type=str, default='dpt_hybrid', choices=['dpt_hybrid', 'dpt_large'], help='DPT model type.')
    parser.add_argument('--blur_head_type', type=str, default='medium_blur_head', 
                        choices=['original_blur_head', 'lightweight_blur_head', 'medium_blur_head'], 
                        help='Type of blur head architecture to use.')
    parser.add_argument('--img_size', type=int, default=384, help='Image size to resize to for DPT input.')
    parser.add_argument('--output_channels', type=int, default=3, help='Number of output channels (must be 3 for bx, by, magnitude).')
    # Training Args
    parser.add_argument('--epochs', type=int, default=50, help='Total number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    # Checkpoint Args
    parser.add_argument('--checkpoint_dir', type=str, default='./dpt_blur_checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--job_name', type=str, default=None, help='Name for this training job. If not provided, will use timestamp.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from (.pth).')
    # Add new argument for direction prediction
    parser.add_argument('--use_direction', action='store_true',
                        help='Whether to use direction prediction in the loss function.')

    args = parser.parse_args()

    # Create job-specific checkpoint directory
    if args.job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.job_name = f"job_{timestamp}"
    
    # Create full checkpoint path with job name
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.job_name)
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    # Set random seed for reproducibility
    random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)

    # --- Setup --- 
    start_epoch = 0
    best_val_loss = float('inf')

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Ensure output channels is 3 for bx, by, magnitude prediction
    if args.output_channels != 3:
        raise ValueError("--output_channels must be 3 for (bx, by, magnitude) vector prediction.")

    # Transforms
    dpt_transform = Compose([
        Resize(args.img_size, args.img_size, resize_target=None, keep_aspect_ratio=True,
               ensure_multiple_of=32, resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])
    target_transform = None

    # Construct full paths from the base dataset directory
    blurred_dir_train = os.path.join(args.dataset_dir, 'train', 'blur')
    gt_dir_train = os.path.join(args.dataset_dir, 'train', 'condition')
    blurred_dir_test = os.path.join(args.dataset_dir, 'val', 'blur')
    gt_dir_test = os.path.join(args.dataset_dir, 'val', 'condition')

    # Datasets
    print("Setting up datasets...")
    try:
        # Create full training dataset
        full_train_dataset = BlurMapDataset(
            blurred_dir_train,
            gt_dir_train,
            transform=dpt_transform,
            target_transform=target_transform,
            crop_size=args.img_size,
            is_train=True,
            random_flip=True
        )
        
        # Split training data into train and validation sets
        train_size = int((1 - args.val_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.split_seed)
        )
        
        # Create test dataset from original validation set
        test_dataset = BlurMapDataset(
            blurred_dir_test,
            gt_dir_test,
            transform=dpt_transform,
            target_transform=target_transform,
            crop_size=args.img_size,
            is_train=False,
            random_flip=False
        )
        
        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        exit()

    # Create dataloaders
    use_pin_memory = True if device.type == 'cuda' else False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=use_pin_memory, 
        collate_fn=collate_fn_skip_none
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=use_pin_memory, 
        collate_fn=collate_fn_skip_none
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=use_pin_memory, 
        collate_fn=collate_fn_skip_none
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Model
    print("Creating model...")
    model = create_dpt_blur_model(
        output_channels=args.output_channels,
        model_type=args.model_type,
        blur_head_type=args.blur_head_type,
        pretrained_weights_path=args.weights,
        freeze_backbone=True
    )
    model.to(device)

    # Loss and Optimizer
    criterion = BlurVectorLoss(
        use_direction=args.use_direction,
        direction_weight=1.0 if args.use_direction else 0.0,
        magnitude_weight=1.0
    )
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # --- Resume from Checkpoint --- 
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load Model state
            try:
                 model.load_state_dict(checkpoint['model_state_dict'])
            except KeyError:
                 print("Warning: Checkpoint missing 'model_state_dict'. Model weights not loaded.")
            except Exception as e:
                 print(f"Error loading model state_dict: {e}")

            # Load Optimizer state
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Manually move optimizer state to device (important if device differs from save time)
                for state in optimizer.state.values():
                     for k, v in state.items():
                         if isinstance(v, torch.Tensor):
                             state[k] = v.to(device)
            except KeyError:
                 print("Warning: Checkpoint missing 'optimizer_state_dict'. Optimizer state not loaded.")
            except Exception as e:
                 print(f"Error loading optimizer state_dict: {e}")

            # Load Scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"Error loading scheduler state_dict: {e}")
            
            # Load Epoch and Best Loss
            try:
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                print(f"Resuming training from Epoch {start_epoch + 1}. Best validation loss so far: {best_val_loss:.6f}")
            except KeyError:
                 print("Warning: Checkpoint missing 'epoch' or 'best_val_loss'. Starting from epoch 0.")
                 start_epoch = 0
                 best_val_loss = float('inf')
            
            # Clean up checkpoint variable
            del checkpoint
            torch.cuda.empty_cache()

        else:
            print(f"Warning: No checkpoint found at '{args.resume}'. Training from scratch.")

    # --- Start Training --- 
    print(f"Starting training from Epoch {start_epoch + 1}...")
    best_val_loss, all_stats = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        start_epoch,
        args.epochs,
        args.checkpoint_dir,
        best_val_loss
    ) 

    # Add test evaluation after training
    print("\nEvaluating on test set...")
    model.eval()
    test_stats = {
        'magnitude_loss': 0.0,
        'total_loss': 0.0,
        'abs_magnitude_error': 0.0,
        'pred_mag_mean': 0.0,
        'pred_mag_std': 0.0,
        'pred_mag_max': 0.0,
        'target_mag_mean': 0.0,
        'target_mag_std': 0.0,
        'target_mag_max': 0.0,
        'relative_error_mean': 0.0,
        'relative_error_std': 0.0,
        'mse_total': 0.0,
        'mse_magnitude': 0.0,
        'psnr': 0.0
    }
    
    # Add direction-related statistics only if using direction
    if args.use_direction:
        test_stats.update({
            'direction_loss': 0.0,
            'cosine_sim_mean': 0.0,
            'angle_diff_mean': 0.0,
            'direction_diff_mean': 0.0,
            'mse_direction': 0.0,
            'mse_bx': 0.0,
            'mse_by': 0.0
        })
    
    test_batch_count = 0
    all_pred_mags = []
    all_target_mags = []
    
    if args.use_direction:
        all_pred_bx = []
        all_pred_by = []
        all_target_bx = []
        all_target_by = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None: continue
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            if outputs.shape[-2:] != targets.shape[-2:]:
                outputs_resized = F_nn.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            else:
                outputs_resized = outputs

            loss, batch_stats = criterion(outputs_resized, targets)
            
            # Store predictions and targets for per-image analysis
            all_pred_mags.append(outputs_resized[:, 2].cpu())
            all_target_mags.append(targets[:, 2].cpu())
            
            if args.use_direction:
                all_pred_bx.append(outputs_resized[:, 0].cpu())
                all_pred_by.append(outputs_resized[:, 1].cpu())
                all_target_bx.append(targets[:, 0].cpu())
                all_target_by.append(targets[:, 1].cpu())
            
            batch_size = inputs.size(0)
            for key in test_stats:
                if key in batch_stats:  # Only update if key exists in batch_stats
                    test_stats[key] += batch_stats[key] * batch_size
            test_batch_count += batch_size

    # Calculate and print average statistics for test set
    if test_batch_count > 0:
        print("\nTest Set Statistics:")
        for key in test_stats:
            test_stats[key] /= test_batch_count
            print(f"{key}: {test_stats[key]:.6f}")
        
        # Calculate per-image statistics for magnitude
        all_pred_mags = torch.cat(all_pred_mags, dim=0)
        all_target_mags = torch.cat(all_target_mags, dim=0)
        
        # Calculate per-image magnitude MSE (averaging over spatial dimensions)
        per_image_mse_mag = torch.mean((all_pred_mags - all_target_mags) ** 2, dim=(1, 2))
        
        print("\nPer-Image Magnitude Statistics:")
        print(f"MSE - Mean: {per_image_mse_mag.mean():.6f}, Std: {per_image_mse_mag.std():.6f}")
        print(f"PSNR: {10 * torch.log10(1.0 / (per_image_mse_mag.mean() + 1e-6)):.2f} dB")
        
        if args.use_direction:
            # Calculate per-image statistics for direction
            all_pred_bx = torch.cat(all_pred_bx, dim=0)
            all_pred_by = torch.cat(all_pred_by, dim=0)
            all_target_bx = torch.cat(all_target_bx, dim=0)
            all_target_by = torch.cat(all_target_by, dim=0)
            
            per_image_mse_bx = torch.mean((all_pred_bx - all_target_bx) ** 2, dim=(1, 2))
            per_image_mse_by = torch.mean((all_pred_by - all_target_by) ** 2, dim=(1, 2))
            
            print("\nPer-Image Direction Statistics:")
            print(f"X MSE - Mean: {per_image_mse_bx.mean():.6f}, Std: {per_image_mse_bx.std():.6f}")
            print(f"Y MSE - Mean: {per_image_mse_by.mean():.6f}, Std: {per_image_mse_by.std():.6f}")
            print(f"X PSNR: {10 * torch.log10(1.0 / (per_image_mse_bx.mean() + 1e-6)):.2f} dB")
            print(f"Y PSNR: {10 * torch.log10(1.0 / (per_image_mse_by.mean() + 1e-6)):.2f} dB")

    print("\nDPT_blur training completed.")
