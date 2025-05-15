import torch
import os
import sys
import argparse

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import directly from DPT_blur
from DPT_blur.process_gopro_dataset import process_gopro_dataset
from DPT_blur.data_loader import BlurMapDataset
from DPT_blur.dpt_lib.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2

def train_mimo_unet_blur_field():
    parser = argparse.ArgumentParser(description="Train MIMO-UNet for blur field prediction")
    parser.add_argument('--gopro_root', type=str, required=True, help='Root directory of GOPRO dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for processed dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    # Process dataset using DPT_blur's function
    process_gopro_dataset(args.gopro_root, args.output_dir)
    
    # Create transforms using DPT_blur's transform classes
    transform = Compose([
        Resize(
            256, 256,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=8,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])
    
    # Create datasets using DPT_blur's dataset class
    train_dataset = BlurMapDataset(
        blurred_dir=os.path.join(args.output_dir, 'train_images'),
        gt_dir=os.path.join(args.output_dir, 'train_gt'),
        transform=transform,
        is_train=True,
        crop_size=256,
        random_flip=True
    )
    
    val_dataset = BlurMapDataset(
        blurred_dir=os.path.join(args.output_dir, 'val_images'),
        gt_dir=os.path.join(args.output_dir, 'val_gt'),
        transform=transform,
        is_train=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize MIMO-UNet model
    # (Your MIMO-UNet model initialization code here)
    
    # Training loop
    # (Your training loop code here)

if __name__ == "__main__":
    train_mimo_unet_blur_field()