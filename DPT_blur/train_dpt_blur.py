import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Compose
import torchvision.transforms.functional as TF
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

# --- Dataset Definition ---

class BlurMapDataset(Dataset):
    """Dataset for loading blurred images and their corresponding blur map ground truth."""
    def __init__(self, blurred_dir, gt_dir, transform=None, target_transform=None):
        self.blurred_dir = blurred_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.target_transform = target_transform

        self.image_files = sorted(glob.glob(os.path.join(blurred_dir, '*.png'))) \
                         + sorted(glob.glob(os.path.join(blurred_dir, '*.jpg'))) \
                         + sorted(glob.glob(os.path.join(blurred_dir, '*.jpeg')))

        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {blurred_dir}")

        print(f"Found {len(self.image_files)} potential image files.")
        # TODO: Consider pre-filtering image_files list here based on existing GT files
        # self.image_files = self._pre_filter_pairs(self.image_files, self.gt_dir)
        # print(f"Found {len(self.image_files)} valid image/GT pairs.")

    # Optional pre-filtering method
    # def _pre_filter_pairs(self, image_files, gt_dir):
    #     valid_files = []
    #     for img_path in image_files:
    #         base_name = os.path.splitext(os.path.basename(img_path))[0]
    #         gt_path = os.path.join(gt_dir, base_name + '.npy')
    #         if os.path.exists(gt_path):
    #             valid_files.append(img_path)
    #     return valid_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(self.gt_dir, base_name + '.npy')

        if not os.path.exists(gt_path):
            # This path should ideally not be hit if pre-filtering is done
            print(f"Warning: GT not found for {img_path} (Index {idx}). Returning None.")
            return None # Signal to filter this out later

        # Load image
        try:
            image = cv2.imread(img_path)
            if image is None: raise IOError(f"imread failed for {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
        except Exception as e:
             print(f"Error loading image {img_path}: {e}. Returning None.")
             return None

        # --- Ground Truth Loading and Processing --- 
        try:
            # TODO: This section is DEPENDENT on the final GT format.
            # Adapt the loading and conversion based on how GT is stored.

            # --- Placeholder Example: Assumes GT is (Mag, Angle_Radians) --- 
            gt_map_mag_angle = np.load(gt_path).astype(np.float32)
            if gt_map_mag_angle.shape[0] != 2:
                raise ValueError(f"Expected GT shape (2, H, W), got {gt_map_mag_angle.shape}")
            
            print(f"DEBUG: Converting GT {base_name}.npy from (Mag, Angle) -> (bx, by)")
            magnitude = gt_map_mag_angle[0, :, :]
            angle_rad = gt_map_mag_angle[1, :, :] # Make sure this is in RADIANS
            bx = magnitude * np.cos(angle_rad)
            by = magnitude * np.sin(angle_rad)
            gt_map_xy = np.stack((bx, by), axis=0)
            # --- End Placeholder Example --- 

            # TODO: If your GT is ALREADY stored as (bx, by), replace the above with:
            # gt_map_xy = np.load(gt_path).astype(np.float32)
            # if gt_map_xy.shape[0] != 2:
            #     raise ValueError(f"Expected GT shape (2, H, W) for (bx, by), got {gt_map_xy.shape}")

        except Exception as e:
             print(f"Error loading/processing ground truth {gt_path}: {e}. Returning None.")
             return None
        # --- End Ground Truth Section ---

        # Apply input image transforms
        if self.transform:
            sample = {"image": image}
            transformed_sample = self.transform(sample)
            image_tensor = transformed_sample["image"]
        else:
            image_tensor = TF.to_tensor(image)

        # Resize GT map (bx, by) to match transformed input size
        gt_tensor = torch.from_numpy(gt_map_xy).unsqueeze(0)
        target_size = image_tensor.shape[1:]
        gt_resized_tensor = TF.interpolate(gt_tensor, size=target_size, mode='bilinear', align_corners=False)
        gt_resized_tensor = gt_resized_tensor.squeeze(0)

        # Apply optional target transforms (e.g., normalization)
        if self.target_transform:
             gt_resized_tensor = self.target_transform(gt_resized_tensor)

        return image_tensor, gt_resized_tensor

# --- Model Creation ---

def create_dpt_blur_model(output_channels=2, model_type="dpt_hybrid", pretrained_weights_path=None):
    """Creates the DPT model, loads pre-trained weights, replaces the head."""

    print(f"Creating DPT model (type: {model_type}) for {output_channels}-channel output.")

    # Determine backbone and features
    if model_type == "dpt_large":
        backbone = "vitl16_384"
        features = 256
    elif model_type == "dpt_hybrid":
        backbone = "vitb_rn50_384"
        features = 256
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Create base DPT model with a dummy head
    model = DPT(head=nn.Identity(), backbone=backbone, features=features, use_bn=True)

    # Load pre-trained weights for the BACKBONE if path is provided
    if pretrained_weights_path:
        print(f"Loading pre-trained BACKBONE weights from: {pretrained_weights_path}")
        if not os.path.exists(pretrained_weights_path):
             raise FileNotFoundError(f"Pretrained weights not found at {pretrained_weights_path}")

        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint

        # Filter out segmentation head and aux layer weights
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("scratch.output_conv.") and not k.startswith("auxlayer."):
                filtered_state_dict[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        print("Pre-trained backbone weights loaded.")
        # Print warnings for unexpected issues during backbone loading
        if unexpected_keys:
             print(f"  Warning: Unexpected keys during backbone weight load: {unexpected_keys}")
        if missing_keys and any(not k.startswith("scratch.output_conv.") for k in missing_keys):
             print(f"  Warning: Missing non-head keys during backbone weight load: {[k for k in missing_keys if not k.startswith('scratch.output_conv.')]}")

    else:
        print("Warning: No pre-trained backbone weights path provided.")

    # Freeze backbone parameters (ensure this runs AFTER potential backbone loading)
    print("Freezing backbone parameters...")
    num_frozen = 0
    for name, param in model.named_parameters():
        if not name.startswith("scratch.output_conv."): # Keep head unfrozen
            param.requires_grad = False
            num_frozen += 1
        else:
             param.requires_grad = True # Ensure head is trainable
             # print(f"  - Parameter unfrozen (part of head): {name}") # Optional: verbose
    print(f"Froze {num_frozen} parameters in the backbone.")

    # Replace the head with a new regression head
    model.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0)
        )
    print(f"Initialized new model head for {output_channels}-channel regression.")

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable head parameters: {trainable_params:,}")

    return model

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
                 outputs_resized = TF.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            else:
                 outputs_resized = outputs

            loss = criterion(outputs_resized, targets)
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
                     outputs_resized = TF.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
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
    parser.add_argument('--blurred_dir_train', type=str, required=True, help='Directory containing blurred training images.')
    parser.add_argument('--gt_dir_train', type=str, required=True, help='Directory containing training ground truth .npy blur maps.')
    parser.add_argument('--blurred_dir_val', type=str, required=True, help='Directory containing blurred validation images.')
    parser.add_argument('--gt_dir_val', type=str, required=True, help='Directory containing validation ground truth .npy blur maps.')
    # Model Args
    parser.add_argument('--weights', type=str, required=True, help='Path to pre-trained DPT segmentation weights (.pt file) for backbone initialization.')
    parser.add_argument('--model_type', type=str, default='dpt_hybrid', choices=['dpt_hybrid', 'dpt_large'], help='DPT model type.')
    parser.add_argument('--img_size', type=int, default=384, help='Image size to resize to for DPT input.')
    parser.add_argument('--output_channels', type=int, default=2, help='Number of output channels (must be 2 for bx, by).')
    # Training Args
    parser.add_argument('--epochs', type=int, default=50, help='Total number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
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

    # Ensure output channels is 2 for bx, by prediction
    if args.output_channels != 2:
        raise ValueError("--output_channels must be 2 for (bx, by) vector prediction.")

    # Transforms
    dpt_transform = Compose([
        Resize(args.img_size, args.img_size, resize_target=None, keep_aspect_ratio=True,
               ensure_multiple_of=32, resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])
    # TODO: Define target_transform if GT needs normalization/scaling
    target_transform = None

    # Datasets
    print("Setting up datasets...")
    try:
        train_dataset_full = BlurMapDataset(args.blurred_dir_train, args.gt_dir_train, transform=dpt_transform, target_transform=target_transform)
        val_dataset_full = BlurMapDataset(args.blurred_dir_val, args.gt_dir_val, transform=dpt_transform, target_transform=target_transform)
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        exit()

    # Create a collate_fn to handle None values returned by __getitem__
    def collate_fn_skip_none(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: return None # Return None if the whole batch is invalid
        return torch.utils.data.dataloader.default_collate(batch)

    # DataLoaders
    train_loader = DataLoader(train_dataset_full, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset_full, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
    print(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader)} batches.")
    # Note: len(loader) gives number of batches. len(loader.dataset) gives original dataset size.

    # Model
    print("Creating model...")
    model = create_dpt_blur_model(
        output_channels=args.output_channels,
        model_type=args.model_type,
        pretrained_weights_path=args.weights # Pass backbone weights path
    )
    model.to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

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