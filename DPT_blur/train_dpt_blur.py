import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.transforms import Compose
import torchvision.transforms.functional as TF
import torch.nn.functional as F_nn
import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm
import random # For dataset filtering example

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

class BlurVectorLoss(nn.Module):
    """
    Loss function for blur vectors with configurable direction prediction.
    When use_direction=True:
        - Handles both direction and magnitude prediction
        - Uses direct vector comparison for direction
        - Uses relative/absolute error for magnitude
    When use_direction=False:
        - Focuses only on magnitude prediction
        - Uses relative/absolute error for magnitude
    """
    def __init__(self, use_direction=False, direction_weight=20.0, magnitude_weight=1.0, eps=1e-6):
        super(BlurVectorLoss, self).__init__()
        self.use_direction = use_direction
        self.direction_weight = direction_weight if use_direction else 0.0
        self.magnitude_weight = magnitude_weight
        self.eps = eps
        
    def forward(self, pred, target):
        # Extract components
        pred_bx, pred_by, pred_mag = pred[:, 0], pred[:, 1], pred[:, 2]
        target_bx, target_by, target_mag = target[:, 0], target[:, 1], target[:, 2]
        
        # Ensure positive magnitude predictions
        pred_mag = torch.abs(pred_mag)
        
        # Calculate magnitude error
        abs_error = torch.abs(pred_mag - target_mag)
        rel_error = abs_error / (target_mag + self.eps)
        
        # Calculate MSE for monitoring
        mse_total = torch.mean((pred - target) ** 2)
        mse_magnitude = torch.mean((pred_mag - target_mag) ** 2)
        mse_direction = torch.mean((pred_bx - target_bx) ** 2 + (pred_by - target_by) ** 2)
        mse_bx = torch.mean((pred_bx - target_bx) ** 2)
        mse_by = torch.mean((pred_by - target_by) ** 2)
        
        # Magnitude loss with adaptive weighting
        magnitude_loss = torch.where(
            target_mag < 0.1,  # For small magnitudes
            rel_error,
            abs_error  # Use absolute error for larger magnitudes
        )
        
        # Weight magnitude loss by target magnitude
        magnitude_loss = (magnitude_loss * target_mag).mean()
        
        # Initialize direction loss and related metrics
        direction_loss = torch.tensor(0.0, device=pred.device)
        cos_sim = torch.tensor(1.0, device=pred.device)
        angle_diff = torch.tensor(0.0, device=pred.device)
        direction_diff = torch.tensor(0.0, device=pred.device)
        
        if self.use_direction:
            # Calculate vector statistics
            pred_norm = torch.sqrt(pred_bx**2 + pred_by**2 + self.eps)
            target_norm = torch.sqrt(target_bx**2 + target_by**2 + self.eps)
            
            # Normalize vectors for direction comparison
            pred_bx_norm = pred_bx / pred_norm
            pred_by_norm = pred_by / pred_norm
            target_bx_norm = target_bx / target_norm
            target_by_norm = target_by / target_norm
            
            # Direction loss using direct vector comparison
            direction_diff = torch.sqrt(
                (pred_bx_norm - target_bx_norm)**2 + 
                (pred_by_norm - target_by_norm)**2 + 
                self.eps
            )
            
            # Weight direction loss by target magnitude
            direction_loss = (direction_diff * target_mag).mean()
            
            # Calculate cosine similarity for monitoring
            cos_sim = (pred_bx_norm * target_bx_norm + pred_by_norm * target_by_norm).clamp(-1.0 + self.eps, 1.0 - self.eps)
            angle_diff = torch.acos(cos_sim)  # Angle difference in radians
        
        # Calculate total loss
        total_loss = self.direction_weight * direction_loss + self.magnitude_weight * magnitude_loss
        
        # Calculate additional statistics
        stats = {
            'magnitude_loss': magnitude_loss.item(),
            'total_loss': total_loss.item(),
            'abs_magnitude_error': abs_error.mean().item(),
            'pred_mag_mean': pred_mag.mean().item(),
            'pred_mag_std': pred_mag.std().item(),
            'pred_mag_max': pred_mag.max().item(),
            'target_mag_mean': target_mag.mean().item(),
            'target_mag_std': target_mag.std().item(),
            'target_mag_max': target_mag.max().item(),
            'relative_error_mean': rel_error.mean().item(),
            'relative_error_std': rel_error.std().item(),
            'mse_total': mse_total.item(),
            'mse_magnitude': mse_magnitude.item(),
            'mse_direction': mse_direction.item(),
            'mse_bx': mse_bx.item(),
            'mse_by': mse_by.item(),
            'psnr': (10 * torch.log10(1.0 / (mse_total + self.eps))).item()
        }
        
        # Add direction-related statistics only if using direction
        if self.use_direction:
            stats.update({
                'direction_loss': direction_loss.item(),
                'cosine_sim_mean': cos_sim.mean().item(),
                'angle_diff_mean': angle_diff.mean().item(),
                'direction_diff_mean': direction_diff.mean().item()
            })
        
        return total_loss, stats

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

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        print(f"\n--- Epoch {current_epoch}/{epochs} ---")

        # --- Training Phase ---
        model.train()
        epoch_stats = {
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
            'mse_direction': 0.0,
            'mse_bx': 0.0,
            'mse_by': 0.0,
            'psnr': 0.0
        }
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {current_epoch} Training")
        batch_count = 0
        for batch_idx, batch in enumerate(pbar_train):
            if batch is None:
                print(f"Warning: Skipping None batch at index {batch_idx}")
                continue
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if outputs.shape[-2:] != targets.shape[-2:]:
                 outputs_resized = F_nn.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
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
                epoch_stats[key] += batch_stats[key] * batch_size
            batch_count += batch_size
            
            # Update progress bar with key metrics
            pbar_train.set_postfix({
                'total': f"{batch_stats['total_loss']:.4f}",
                'mag': f"{batch_stats['magnitude_loss']:.4f}",
                'mse': f"{batch_stats['mse_total']:.4f}"
            })

        # Calculate and print average statistics for training
        if batch_count > 0:
            print("\nTraining Statistics:")
            for key in epoch_stats:
                epoch_stats[key] /= batch_count
                print(f"{key}: {epoch_stats[key]:.6f}")
        all_stats['train'].append(epoch_stats)

        # --- Validation Phase ---
        model.eval()
        val_stats = {k: 0.0 for k in epoch_stats}
        pbar_val = tqdm(val_loader, desc=f"Epoch {current_epoch} Validation")
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar_val):
                if batch is None: continue
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                if outputs.shape[-2:] != targets.shape[-2:]:
                     outputs_resized = F_nn.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                else:
                     outputs_resized = outputs

                loss, batch_stats = criterion(outputs_resized, targets)
                
                if not torch.isnan(loss):
                    batch_size = inputs.size(0)
                    for key in val_stats:
                        val_stats[key] += batch_stats[key] * batch_size
                    val_batch_count += batch_size
                    
                    pbar_val.set_postfix({
                        'total': f"{batch_stats['total_loss']:.4f}",
                        'mag': f"{batch_stats['magnitude_loss']:.4f}",
                        'mse': f"{batch_stats['mse_total']:.4f}"
                    })

        # Calculate and print average statistics for validation
        if val_batch_count > 0:
            print("\nValidation Statistics:")
            for key in val_stats:
                val_stats[key] /= val_batch_count
                print(f"{key}: {val_stats[key]:.6f}")
        all_stats['val'].append(val_stats)

        # Update learning rate scheduler using total loss
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(val_stats['total_loss'])
            else:
                 scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                 print(f"Learning rate changed to {new_lr:.8f}")

        # --- Checkpoint Saving --- 
        is_best = val_stats['total_loss'] < best_val_loss
        if is_best:
            print(f"Validation loss improved ({best_val_loss:.6f} --> {val_stats['total_loss']:.6f}).")
            best_val_loss = val_stats['total_loss']

        # Save checkpoint with additional statistics
        checkpoint_data = {
            'epoch': current_epoch,
            'head_state_dict': model.scratch.output_conv.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_stats': epoch_stats,
            'val_stats': val_stats
        }
        if scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'dpt_blur_latest.pth')
        try:
            torch.save(checkpoint_data, latest_checkpoint_path)
        except Exception as e:
            print(f"Error saving latest checkpoint: {e}")

        # Save best checkpoint if loss improved
        if is_best:
            best_checkpoint_path = os.path.join(checkpoint_dir, 'dpt_blur_best.pth')
            try:
                torch.save(checkpoint_data, best_checkpoint_path)
                print(f"Saved best model head state to {best_checkpoint_path}")
            except Exception as e:
                print(f"Error saving best checkpoint: {e}")

    print("\nTraining finished.")
    print(f"Final best validation loss: {best_val_loss:.6f}")
    
    # Print final statistics summary
    print("\nFinal Statistics Summary:")
    print("Training:")
    for key in all_stats['train'][-1]:
        print(f"{key}: {all_stats['train'][-1][key]:.6f}")
    print("\nValidation:")
    for key in all_stats['val'][-1]:
        print(f"{key}: {all_stats['val'][-1][key]:.6f}")

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
    parser.add_argument('--blur_head_type', type=str, default='enhanced_blur_head', 
                        choices=['enhanced_blur_head', 'original_blur_head', 'lightweight_blur_head', 'medium_blur_head'], 
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
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from (.pth).')
    # Add new argument for direction prediction
    parser.add_argument('--use_direction', action='store_true',
                        help='Whether to use direction prediction in the loss function.')

    args = parser.parse_args()

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
        direction_weight=20.0 if args.use_direction else 0.0,
        magnitude_weight=1.0
    )
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # --- Resume from Checkpoint --- 
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load Head state
            try:
                 model.scratch.output_conv.load_state_dict(checkpoint['head_state_dict'])
            except KeyError:
                 print("Warning: Checkpoint missing 'head_state_dict'. Head weights not loaded.")
            except Exception as e:
                 print(f"Error loading head state_dict: {e}")

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
    train_model(
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
        'mse_magnitude': 0.0,
        'psnr_magnitude': 0.0
    }
    
    # Add direction-related statistics if using direction
    if args.use_direction:
        test_stats.update({
            'direction_loss': 0.0,
            'cosine_sim_mean': 0.0,
            'angle_diff_mean': 0.0,
            'direction_diff_mean': 0.0,
            'mse_direction': 0.0,
            'mse_bx': 0.0,
            'mse_by': 0.0,
            'psnr_direction': 0.0
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
