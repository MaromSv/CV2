import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader # Removed Subset for simplicity in dummy script
from torchvision.transforms import Compose
import torchvision.transforms.functional as TF
import torch.nn.functional as F # Import torch.nn.functional
import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm
import random

# Import DPT model and transforms
try:
    from dpt_lib.models import DPT
    from dpt_lib.transforms import Resize, NormalizeImage, PrepareForNet
    # Interpolate might not be needed if output matches target directly from model head
except ImportError as e:
    print("Error: Could not import local DPT library (dpt_lib). Make sure it exists.")
    raise

# --- Import Model Creation Utility ---
from model_utils import create_dpt_blur_model

# --- Top-level collate_fn for multiprocessing compatibility ---
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Dummy Ground Truth Dataset ---
class DummyGTDataset(Dataset):
    """
    Dataset for loading blurred images and generating dummy ground truth.
    Includes optional random cropping for training augmentation.
    """
    def __init__(self, blurred_dir, transform=None, crop_size=None, output_channels=3, random_flip_prob=0.0):
        """
        Args:
            blurred_dir (str): Directory containing blurred input images.
            transform (callable, optional): Transform for the input image.
            crop_size (int, optional): Desired size for random cropping.
            output_channels (int): Number of channels for the dummy GT.
            random_flip_prob (float): Probability of applying a random horizontal flip.
        """
        self.blurred_dir = blurred_dir
        self.transform = transform
        self.crop_size = crop_size
        self.output_channels = output_channels
        self.random_flip_prob = random_flip_prob

        self.image_files = sorted(glob.glob(os.path.join(blurred_dir, '*.png'))) \
                         + sorted(glob.glob(os.path.join(blurred_dir, '*.jpg'))) \
                         + sorted(glob.glob(os.path.join(blurred_dir, '*.jpeg')))
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {blurred_dir}")
        print(f"Found {len(self.image_files)} image files in {blurred_dir} for DummyGTDataset.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = cv2.imread(img_path) # HWC, BGR
            if image is None: raise IOError(f"imread failed for {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # HWC, RGB
            image = image.astype(np.float32)
        except Exception as e:
             print(f"Error loading image {img_path}: {e}. Returning None.")
             return None

        # Random Cropping
        if self.crop_size:
            H_orig, W_orig = image.shape[:2]
            if H_orig < self.crop_size or W_orig < self.crop_size:
                # If image is smaller than crop_size, resize it to crop_size
                # This is a simple strategy for a dummy loader; more sophisticated handling might be needed
                image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                # print(f"Warning: Image {os.path.basename(img_path)} ({H_orig}x{W_orig}) smaller than crop size ({self.crop_size}). Resized.")
            else:
                top = random.randint(0, H_orig - self.crop_size)
                left = random.randint(0, W_orig - self.crop_size)
                image = image[top:top+self.crop_size, left:left+self.crop_size, :]

        # Random Horizontal Flip
        if random.random() < self.random_flip_prob:
            image = cv2.flip(image, 1) # HWC

        # Apply Input Image Transforms (e.g., DPT normalization, ToTensor)
        if self.transform:
            sample = {"image": image} # Transform expects a dict
            transformed_sample = self.transform(sample)
            image_tensor = transformed_sample["image"]
        else:
            # Basic fallback: Convert to CHW tensor
            image_tensor = TF.to_tensor(image)

        # Create Dummy Ground Truth Tensor
        # GT shape should match the processed image tensor's spatial dimensions
        _, H_processed, W_processed = image_tensor.shape
        dummy_gt_tensor = torch.zeros((self.output_channels, H_processed, W_processed), dtype=torch.float32)

        return image_tensor, dummy_gt_tensor

# --- Modified Training Function ---
def train_model(model, train_loader, criterion, optimizer, scheduler, device, start_epoch, epochs, checkpoint_dir):
    """Main training loop with checkpoint saving (no validation)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    # best_val_loss is removed as there's no validation

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        print(f"\n--- Epoch {current_epoch}/{epochs} ---")

        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {current_epoch} Training")
        actual_processed_train = 0

        for batch_idx, batch in enumerate(pbar_train):
            if batch is None:
                print(f"Warning: Skipping None batch at index {batch_idx}")
                continue
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if outputs.shape[-2:] != targets.shape[-2:]:
                 outputs_resized = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            else:
                 outputs_resized = outputs

            loss = criterion(outputs_resized.contiguous(), targets.contiguous())
            if torch.isnan(loss):
                 print(f"Warning: NaN loss encountered at Epoch {current_epoch}, Batch {batch_idx}. Skipping batch.")
                 optimizer.zero_grad()
                 continue

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            actual_processed_train += inputs.size(0)
            pbar_train.set_postfix({'loss': f'{loss.item():.4f}'})

        if actual_processed_train == 0:
            avg_train_loss = 0
            print("Warning: No training samples processed in epoch.")
        else:
            avg_train_loss = train_loss / actual_processed_train
        print(f"Epoch {current_epoch} Average Training Loss: {avg_train_loss:.6f}")

        # Skip Validation Phase

        # Update learning rate scheduler (if not ReduceLROnPlateau or if stepping per epoch)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                 print(f"Learning rate changed to {new_lr:.8f}")
        elif scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            print("Note: ReduceLROnPlateau scheduler normally steps based on validation loss. It will not step here.")


        # --- Checkpoint Saving (Latest Only) ---
        checkpoint_data = {
            'epoch': current_epoch,
            'head_state_dict': model.scratch.output_conv.state_dict(), # Assuming head is in model.scratch.output_conv
            'optimizer_state_dict': optimizer.state_dict(),
            # 'best_val_loss': best_val_loss, # Removed
        }
        if scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        latest_checkpoint_path = os.path.join(checkpoint_dir, 'dummy_cpt_latest.pth')
        try:
            torch.save(checkpoint_data, latest_checkpoint_path)
            print(f"Saved latest checkpoint to {latest_checkpoint_path}")
        except Exception as e:
            print(f"Error saving latest checkpoint: {e}")

    print("\nTraining finished.")
    # print(f"Final best validation loss: {best_val_loss:.6f}") # Removed

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fine-tuning DPT model with DUMMY ground truth for blur map prediction.")
    # Data Args
    parser.add_argument('--blurred_dir_train', type=str, default='data/GOPRO_Large/train/GOPR0372_07_00/blur', required=False, help='Directory containing blurred training images.')
    # GT dirs are no longer required for this dummy script
    # Model Args
    parser.add_argument('--weights', type=str, default='weights/dpt_large-ade20k-b12dca68.pt', required=False, help='Path to pre-trained DPT segmentation weights (.pt file) for backbone initialization.')
    parser.add_argument('--model_type', type=str, default='dpt_large', choices=['dpt_hybrid', 'dpt_large'], help='DPT model type.')
    parser.add_argument('--img_size', type=int, default=384, help='Image crop size and DPT input size.')
    parser.add_argument('--output_channels', type=int, default=3, help='Number of output channels (must be 3 for bx, by, magnitude).')
    parser.add_argument('--random_flip_prob', type=float, default=0.0, help='Probability for random horizontal flip in DummyGTDataset.')
    # Training Args
    parser.add_argument('--epochs', type=int, default=10, help='Total number of training epochs (reduced for dummy test).')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training (reduced for dummy test).')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader.')
    # Checkpoint Args
    parser.add_argument('--checkpoint_dir', type=str, default='./dummy_checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from (.pth).')

    args = parser.parse_args()

    start_epoch = 0
    # best_val_loss = float('inf') # Removed

    # --- Device Selection (CUDA > MPS > CPU) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.output_channels != 3:
        raise ValueError("--output_channels must be 3 for (bx, by, magnitude) vector prediction.")

    # Transforms (similar to original training script)
    dpt_transform = Compose([
        Resize(args.img_size, args.img_size, resize_target=None, keep_aspect_ratio=True,
               ensure_multiple_of=32, resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])
    # target_transform is not needed for dummy GT

    # Datasets
    print("Setting up dummy GT dataset...")
    try:
        train_dataset = DummyGTDataset(
            args.blurred_dir_train,
            transform=dpt_transform,
            crop_size=args.img_size,
            output_channels=args.output_channels,
            random_flip_prob=args.random_flip_prob
        )
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        exit()

    # Determine pin_memory based on device
    use_pin_memory = True if device.type == 'cuda' else False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_pin_memory, collate_fn=collate_fn_skip_none)
    print(f"Train loader with dummy GT: {len(train_loader)} batches. pin_memory={use_pin_memory}")
    
    val_loader = None # No validation

    print("Creating model...")
    model = create_dpt_blur_model(
        output_channels=args.output_channels,
        model_type=args.model_type,
        pretrained_weights_path=args.weights,
        freeze_backbone=True # Keep this True for initial tests
    )
    model.to(device)

    criterion = nn.MSELoss() # Or any other suitable loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # Scheduler: ReduceLROnPlateau needs validation, so use a simpler one or none for dummy script
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Example: steps every 10 epochs
    scheduler = None # Or keep ReduceLROnPlateau and note it won't step without manual intervention

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            try:
                model.scratch.output_conv.load_state_dict(checkpoint['head_state_dict'])
            except KeyError:
                print("Warning: Checkpoint missing 'head_state_dict'. Head weights not loaded.")
            except Exception as e:
                print(f"Error loading head state_dict: {e}")

            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except KeyError:
                print("Warning: Checkpoint missing 'optimizer_state_dict'. Optimizer state not loaded.")
            except Exception as e:
                print(f"Error loading optimizer state_dict: {e}")
            
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                 try:
                     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 except Exception as e:
                     print(f"Error loading scheduler state_dict: {e}")
            
            try:
                start_epoch = checkpoint['epoch']
                # best_val_loss = checkpoint.get('best_val_loss', float('inf')) # No longer strictly needed
                print(f"Resuming training from Epoch {start_epoch + 1}.")
            except KeyError:
                print("Warning: Checkpoint missing 'epoch'. Starting from epoch 0.")
                start_epoch = 0
            
            del checkpoint
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        else:
            print(f"Warning: No checkpoint found at '{args.resume}'. Training from scratch.")

    print(f"Starting dummy GT training from Epoch {start_epoch + 1}...")
    train_model(
        model,
        train_loader,
        # val_loader is now omitted from this call
        criterion,
        optimizer,
        scheduler,
        device,
        start_epoch,
        args.epochs,
        args.checkpoint_dir
        # best_val_loss is also omitted
    ) 