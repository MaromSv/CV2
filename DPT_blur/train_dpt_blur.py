import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
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
    # from DPT.dpt.models import DPT # Old
    # from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet # Old
    # from DPT.dpt.blocks import Interpolate # Old
    from dpt_lib.models import DPT # New
    from dpt_lib.transforms import Resize, NormalizeImage, PrepareForNet # New
    from dpt_lib.blocks import Interpolate # New
except ImportError as e:
    print("Error: Could not import local DPT library (dpt_lib). Make sure it exists in the same directory as the script.")

# --- Import Dataset Class ---
from data_loader import BlurMapDataset

# --- Import Model Creation Utility ---
from model_utils import create_dpt_blur_model

# --- Define collate_fn at the top level ---
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None # Return None if the whole batch is invalid
    return torch.utils.data.dataloader.default_collate(batch)

# --- Training Function ---

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, start_epoch, epochs, checkpoint_dir, best_val_loss):
    """Main training loop with checkpoint saving."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        print(f"\n--- Epoch {current_epoch}/{epochs} ---")

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {current_epoch} Training")
        for batch_idx, batch in enumerate(pbar_train):
            if batch is None: # Skip if dataset loader returned None
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

            loss = criterion(outputs_resized.contiguous(), targets.contiguous())
            if torch.isnan(loss):
                 print(f"Warning: NaN loss encountered at Epoch {current_epoch}, Batch {batch_idx}. Skipping batch.")
                 optimizer.zero_grad() # Clear potentially bad gradients
                 continue # Skip optimizer step and loss accumulation for this batch

            loss.backward()
            # Optional: Gradient clipping if needed
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            pbar_train.set_postfix({'loss': f'{loss.item():.4f}'})

        # Adjust for potentially skipped batches if dataset size is used for averaging
        num_train_samples = len(train_loader.sampler) if isinstance(train_loader.sampler, torch.utils.data.Sampler) else len(train_loader.dataset)
        # Filter out None items when calculating actual processed items if SubsetRandomSampler not used effectively
        # A more robust count method might be needed depending on dataloader/sampler
        actual_processed_train = sum(b[0].size(0) for b in train_loader if b is not None) # Example count
        if actual_processed_train == 0: 
             avg_train_loss = 0
             print("Warning: No training samples processed in epoch.")
        else: 
             avg_train_loss = train_loss / actual_processed_train
        print(f"Epoch {current_epoch} Average Training Loss: {avg_train_loss:.6f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {current_epoch} Validation")
        actual_processed_val = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar_val):
                if batch is None: continue
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                if outputs.shape[-2:] != targets.shape[-2:]:
                     print(f"Warning: Output shape {outputs.shape[-2:]} does not match target shape {targets.shape[-2:]}. Interpolating outputs.")
                     outputs_resized = F_nn.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                else:
                     outputs_resized = outputs

                loss = criterion(outputs_resized, targets)
                if not torch.isnan(loss):
                    val_loss += loss.item() * inputs.size(0)
                    actual_processed_val += inputs.size(0)
                    pbar_val.set_postfix({'loss': f'{loss.item():.4f}'})
                else:
                    print(f"Warning: NaN validation loss encountered at Epoch {current_epoch}, Batch {batch_idx}. Skipping batch.")

        if actual_processed_val == 0:
             avg_val_loss = float('inf') # Or handle as error
             print("Warning: No validation samples processed in epoch.")
        else:
             avg_val_loss = val_loss / actual_processed_val
        print(f"Epoch {current_epoch} Average Validation Loss: {avg_val_loss:.6f}")

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(avg_val_loss)
            else:
                 scheduler.step() # For schedulers that step per epoch
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                 print(f"Learning rate changed to {new_lr:.8f}")

        # --- Checkpoint Saving --- 
        is_best = avg_val_loss < best_val_loss
        if is_best:
            print(f"Validation loss improved ({best_val_loss:.6f} --> {avg_val_loss:.6f}).")
            best_val_loss = avg_val_loss

        # Prepare checkpoint dictionary
        checkpoint_data = {
            'epoch': current_epoch,
            'head_state_dict': model.scratch.output_conv.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'dpt_blur_latest.pth')
        try:
            torch.save(checkpoint_data, latest_checkpoint_path)
            # print(f"Saved latest checkpoint to {latest_checkpoint_path}")
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

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DPT model for blur map prediction.")
    # Data Args
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Base directory of the restructured dataset (e.g., data/dataset_DPT_blur/), \
                              which should contain train/blur, train/condition, val/blur, val/condition subdirectories.')
    # Model Args
    parser.add_argument('--weights', type=str, default='weights/dpt_large-ade20k-b12dca68.pt', help='Path to pre-trained DPT segmentation weights (.pt file) for backbone initialization.')
    parser.add_argument('--model_type', type=str, default='dpt_large', choices=['dpt_hybrid', 'dpt_large'], help='DPT model type.')
    parser.add_argument('--img_size', type=int, default=384, help='Image size to resize to for DPT input.')
    parser.add_argument('--output_channels', type=int, default=3, help='Number of output channels (must be 3 for bx, by, magnitude).')
    # Training Args
    parser.add_argument('--epochs', type=int, default=50, help='Total number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    # Checkpoint Args
    parser.add_argument('--checkpoint_dir', type=str, default='./dpt_blur_checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from (.pth).')

    args = parser.parse_args()

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
    # TODO: Define target_transform if GT needs normalization/scaling
    target_transform = None

    # Construct full paths from the base dataset directory
    blurred_dir_train = os.path.join(args.dataset_dir, 'train', 'blur')
    gt_dir_train = os.path.join(args.dataset_dir, 'train', 'condition')
    blurred_dir_val = os.path.join(args.dataset_dir, 'val', 'blur')
    gt_dir_val = os.path.join(args.dataset_dir, 'val', 'condition')

    # Datasets
    print("Setting up datasets...")
    try:
        train_dataset_full = BlurMapDataset(
            blurred_dir_train,
            gt_dir_train,
            transform=dpt_transform,
            target_transform=target_transform,
            crop_size=args.img_size,
            is_train=True,
            random_flip=True
        )
        val_dataset_full = BlurMapDataset(
            blurred_dir_val,
            gt_dir_val,
            transform=dpt_transform,
            target_transform=target_transform,
            crop_size=args.img_size,
            is_train=False,
            random_flip=False
        )
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        exit()

    # Create a collate_fn to handle None values returned by __getitem__
    # collate_fn_skip_none is now defined at the top level

    # DataLoaders
    use_pin_memory = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_dataset_full, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_pin_memory, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset_full, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_pin_memory, collate_fn=collate_fn_skip_none)
    print(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader)} batches.")
    # Note: len(loader) gives number of batches. len(loader.dataset) gives original dataset size.

    # Model
    print("Creating model...")
    model = create_dpt_blur_model(
        output_channels=args.output_channels,
        model_type=args.model_type,
        pretrained_weights_path=args.weights, # Pass backbone weights path
        freeze_backbone=True # Explicitly set freeze_backbone, can be an arg if needed
    )
    model.to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

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
                start_epoch = checkpoint['epoch'] # Checkpoint saved epoch *completed*
                best_val_loss = checkpoint['best_val_loss']
                print(f"Resuming training from Epoch {start_epoch + 1}. Best validation loss so far: {best_val_loss:.6f}")
            except KeyError:
                 print("Warning: Checkpoint missing 'epoch' or 'best_val_loss'. Starting from epoch 0.")
                 start_epoch = 0 # Reset if keys are missing
                 best_val_loss = float('inf')
            
            # Clean up checkpoint variable
            del checkpoint
            torch.cuda.empty_cache() # Clear memory if CUDA is used

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
        start_epoch, # Pass the starting epoch (0 if not resuming)
        args.epochs,
        args.checkpoint_dir,
        best_val_loss # Pass the best loss found so far
    ) 
