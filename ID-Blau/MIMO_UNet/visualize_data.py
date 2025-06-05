from torchvision import transforms
import os
import sys
import time
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import math  # Add this import for math.log10
from tqdm import tqdm
import seaborn as sns
import train_blur_field

# Add grandparent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))  # MIMO_UNet directory
parent_dir = os.path.dirname(current_dir)                 # ID-Blau directory
grandparent_dir = os.path.dirname(parent_dir)             # CODE directory containing both ID-Blau and DPT_blur
sys.path.append(grandparent_dir)

# Print paths for debugging
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Grandparent directory: {grandparent_dir}")
print(f"Python path: {sys.path}")

# Import from DPT_blur
try:
    from DPT_blur.data_loader import BlurMapDataset
    from DPT_blur.visualize_blur_map import visualize_blur_field_with_legend
    print("Successfully imported DPT_blur modules")
except ImportError as e:
    print(f"Error importing DPT_blur modules: {e}")
    print("Please make sure DPT_blur is in the same directory as ID-Blau")
    sys.exit(1)

# Import MIMO-UNet model and custom loss
from MIMOUNet import build_MIMOUnet_net
from blur_losses import MultiScaleBlurFieldLoss, BlurFieldLoss, CharbonnierLoss


def parse_args():
    parser = argparse.ArgumentParser(description='vis data')

    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size for training')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use')
    parser.add_argument('--max_val_samples', type=int, default=None, help='Maximum number of validation samples to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for training')
    parser.add_argument('--model_name', type=str, default='MIMO-UNetPlus', help='Model name')
    parser.add_argument('--is_dir', action='store_true', default=False, help='If using direction loss')
    parser.add_argument('--lambda_dir', type=float, default=1.0, help='Lambda for direction loss')
    parser.add_argument('--lambda_mag', type=float, default=1.0, help='Lambda for magnitude loss')
    parser.add_argument('--lambda_mse', type=float, default=1.0, help='Lambda for MSE loss')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='Lambda for L1 loss')
    parser.add_argument('--loss_mode', type=str, default='default', help='Loss mode')
    parser.add_argument('--save_model', action='store_true', default=False, help='Save model')
    return parser.parse_args()

def load_data(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    class MIMONormalize:
        def __call__(self, sample):
            # Convert image to tensor if it's not already
            if not isinstance(sample["image"], torch.Tensor):
                # First make sure image is in [0, 255] range
                image_np = sample["image"].astype(np.float32)
                
                # Debug original range
                # print(f"Original image range: {image_np.min():.4f} to {image_np.max():.4f}")
                
                # Manual conversion to tensor with proper normalization
                # 1. Convert to [0, 1] range
                image_np = image_np / 255.0
                # print(f"After division by 255: {image_np.min():.4f} to {image_np.max():.4f}")
                
                # 2. Convert to tensor (HWC -> CHW)
                image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float()
                # print(f"After tensor conversion: {image_tensor.min().item():.4f} to {image_tensor.max().item():.4f}")
                
                sample["image"] = image_tensor
        
            # Normalize to [-0.5, 0.5] range (standard for MIMO-UNet)
            sample["image"] = sample["image"] - 0.5
            # print(f"After normalization to [-0.5, 0.5]: {sample['image'].min().item():.4f} to {sample['image'].max().item():.4f}")
            
            return sample
    
    transform = transforms.Compose([
        MIMONormalize(),
    ])

    train_dataset = BlurMapDataset(
        blurred_dir=os.path.join(args.train_dir, 'blur'),
        gt_dir=os.path.join(args.train_dir, 'condition'),
        transform=transform,  # Apply MIMO-specific normalization
        crop_size=args.crop_size,
        is_train=True,
    )

    val_dataset = BlurMapDataset(
        blurred_dir=os.path.join(args.val_dir, 'blur'),
        gt_dir=os.path.join(args.val_dir, 'condition'),
        transform=transform,  # Same normalization
        crop_size=args.crop_size,
        is_train=False,
    )

    # Split validation dataset into validation and test sets
    val_size = len(val_dataset)
    test_size = val_size // 2
    val_size = val_size - test_size
    
    from torch.utils.data import random_split
    val_dataset, test_dataset = random_split(
        val_dataset, 
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    logging.info(f"Split validation data: {val_size} validation samples, {test_size} test samples")
    
    # Limit dataset size if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_dataset):
        logging.info(f"Limiting training dataset to {args.max_train_samples} samples (from {len(train_dataset)})")
        # Create a subset of the dataset
        from torch.utils.data import Subset
        indices = list(range(args.max_train_samples))
        train_dataset = Subset(train_dataset, indices)
    
    # Limit validation dataset size if specified
    if args.max_val_samples is not None and args.max_val_samples < len(val_dataset):
        logging.info(f"Limiting validation dataset to {args.max_val_samples} samples (from {len(val_dataset)})")
        from torch.utils.data import Subset
        indices = list(range(args.max_val_samples))
        val_dataset = Subset(val_dataset, indices)
    
    # Log dataset sizes
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    # Log sample dimensions from each dataset
    logging.info("Dataset sample dimensions:")
    try:
        # Get a sample from train dataset
        train_sample = train_dataset[0]
        if isinstance(train_sample, dict):
            logging.info(f"Train sample - Input: {train_sample['blur'].shape}, Blur field: {train_sample['blur_field'].shape}")
        elif isinstance(train_sample, (list, tuple)) and len(train_sample) >= 2:
            logging.info(f"Train sample - Input: {train_sample[0].shape}, Blur field: {train_sample[1].shape}")
        
        # Get a sample from validation dataset
        val_sample = val_dataset[0]
        if isinstance(val_sample, dict):
            logging.info(f"Val sample - Input: {val_sample['blur'].shape}, Blur field: {val_sample['blur_field'].shape}")
        elif isinstance(val_sample, (list, tuple)) and len(val_sample) >= 2:
            logging.info(f"Val sample - Input: {val_sample[0].shape}, Blur field: {val_sample[1].shape}")
        
        # Get a sample from test dataset
        test_sample = test_dataset[0]
        if isinstance(test_sample, dict):
            logging.info(f"Test sample - Input: {test_sample['blur'].shape}, Blur field: {test_sample['blur_field'].shape}")
        elif isinstance(test_sample, (list, tuple)) and len(test_sample) >= 2:
            logging.info(f"Test sample - Input: {test_sample[0].shape}, Blur field: {test_sample[1].shape}")
    except Exception as e:
        logging.warning(f"Error getting sample dimensions: {e}")

    
    # Analyze and log dataset statistics
    try:
        train_stats = analyze_dataset_statistics(train_dataset, "train")
        val_stats = analyze_dataset_statistics(val_dataset, "val")
        
        # Log training dataset statistics
        logging.info("Training dataset blur field statistics:")
        logging.info(f"  bx (cos): min={train_stats['bx']['min']:.4f}, max={train_stats['bx']['max']:.4f}, mean={train_stats['bx']['mean']:.4f}, std={train_stats['bx']['std']:.4f}")
        logging.info(f"  by (sin): min={train_stats['by']['min']:.4f}, max={train_stats['by']['max']:.4f}, mean={train_stats['by']['mean']:.4f}, std={train_stats['by']['std']:.4f}")
        logging.info(f"  magnitude: min={train_stats['magnitude']['min']:.4f}, max={train_stats['magnitude']['max']:.4f}, mean={train_stats['magnitude']['mean']:.4f}, std={train_stats['magnitude']['std']:.4f}")
        
        # Log validation dataset statistics
        logging.info("Validation dataset blur field statistics:")
        logging.info(f"  bx (cos): min={val_stats['bx']['min']:.4f}, max={val_stats['bx']['max']:.4f}, mean={val_stats['bx']['mean']:.4f}, std={val_stats['bx']['std']:.4f}")
        logging.info(f"  by (sin): min={val_stats['by']['min']:.4f}, max={val_stats['by']['max']:.4f}, mean={val_stats['by']['mean']:.4f}, std={val_stats['by']['std']:.4f}")
        logging.info(f"  magnitude: min={val_stats['magnitude']['min']:.4f}, max={val_stats['magnitude']['max']:.4f}, mean={val_stats['magnitude']['mean']:.4f}, std={val_stats['magnitude']['std']:.4f}")
    except Exception as e:
        logging.warning(f"Could not analyze dataset statistics: {e}")
    
    # Select fixed validation samples for consistent visualization
    # Use a time-based seed for reproducibility within this run
    val_sample_seed = 42  # Use current time as seed
    logging.info(f"Using validation sample seed: {val_sample_seed}")
    
    # Save the seed for potential future reference
    with open(os.path.join(args.output_dir, 'val_sample_seed.txt'), 'w') as f:
        f.write(str(val_sample_seed))
    
    # Use the seed to select fixed validation samples
    random.seed(val_sample_seed)
    val_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
    random.seed()  # Reset the seed for other random operations
    
    # Create a fixed validation dataset with just these samples
    from torch.utils.data import Subset
    fixed_val_dataset = Subset(val_dataset, val_indices)
    
    # Create a dataloader for the fixed validation samples
    fixed_val_loader = DataLoader(
        fixed_val_dataset,
        batch_size=1,  # Process one sample at a time for visualization
        shuffle=False,
        num_workers=1
    )
    
    # Visualize dataset samples
    # logging.info("Visualizing dataset samples...")
    # visualize_dataset_samples(train_dataset, args.output_dir, "train", num_samples=10)
    # visualize_dataset_samples(val_dataset, args.output_dir, "val", num_samples=10)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if args.save_model:
        # Create model
        logging.info(f"Creating {args.model_name} model...")
        model = build_MIMOUnet_net(
            model_name=args.model_name
        )

        model_save_path = os.path.join(os.path.dirname(args.output_dir), 'model_init.pth')
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Initial model weights saved to {model_save_path}")

    else:
        logging.info(f"Loading model from {os.path.dirname(args.output_dir)}/model_init.pth")
        model = build_MIMOUnet_net(model_name=args.model_name)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(args.output_dir), 'model_init.pth')))
    
    # The model already has output layers with 3 channels
    # No need to modify the architecture
    
    model.to(device)
    logging.info(f"Model created: {args.model_name}")

    return model, fixed_val_loader, device

def display_data(args, model, fixed_val_loader, device):
    logging.info(f"Creating validation grid visualization...")
    grid_path = train_blur_field.save_validation_grid(model, fixed_val_loader, 1, args.output_dir, device, args)
    logging.info(f"Saved validation grid to {grid_path}")

if __name__ == '__main__':
    args = parse_args()
    model, fixed_val_loader, device = load_data(args)
    display_data(args, model, fixed_val_loader, device)