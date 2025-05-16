import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

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
from blur_losses import BlurFieldLoss, CharbonnierLoss

def train_model(args):
    # Set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # Create datasets and dataloaders
    train_dataset = BlurMapDataset(
        root_dir=args.train_dir,
        transform=None,  # Add transforms if needed
        crop_size=args.crop_size
    )
    
    # Limit dataset size if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_dataset):
        logging.info(f"Limiting training dataset to {args.max_train_samples} samples (from {len(train_dataset)})")
        # Create a subset of the dataset
        from torch.utils.data import Subset
        indices = list(range(args.max_train_samples))
        train_dataset = Subset(train_dataset, indices)
    
    val_dataset = BlurMapDataset(
        root_dir=args.val_dir,
        transform=None,
        crop_size=args.crop_size
    )
    
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
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    model = build_MIMOUnet_net(
        model_name=args.model_name  # Use the model name from arguments
    )
    model = model.to(device)
    logging.info(f"Model created: {model.__class__.__name__}")
    
    # Define loss function
    criterion = BlurFieldLoss(
        lambda_dir=args.lambda_dir,
        lambda_mag=args.lambda_mag
    )
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Initialize tracking variables
    start_epoch = 0
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    val_psnrs = []
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint['best_val_loss']
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            val_psnrs = checkpoint.get('val_psnrs', [])
            logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            logging.error(f"No checkpoint found at {args.resume}")
    
    # Training loop
    logging.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as pbar:
            for batch_idx, sample in enumerate(pbar):
                # Get data
                blur_img = sample['blur'].to(device)
                blur_field = sample['blur_field'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(blur_img)
                
                # Handle multi-scale outputs if present
                if isinstance(outputs, list):
                    # Use the highest resolution output
                    pred = outputs[-1]
                else:
                    pred = outputs
                
                # Compute loss
                loss = criterion(pred, blur_field)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Log batch loss periodically
                if batch_idx % 10 == 0:
                    writer.add_scalar('Loss/train_batch', loss.item(), 
                                     epoch * len(train_loader) + batch_idx)
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]") as pbar:
                for batch_idx, sample in enumerate(pbar):
                    # Get data
                    blur_img = sample['blur'].to(device)
                    blur_field = sample['blur_field'].to(device)
                    
                    # Forward pass
                    outputs = model(blur_img)
                    
                    # Handle multi-scale outputs if present
                    if isinstance(outputs, list):
                        # Use the highest resolution output
                        pred = outputs[-1]
                    else:
                        pred = outputs
                    
                    # Compute loss
                    loss = criterion(pred, blur_field)
                    
                    # Compute PSNR for magnitude channel
                    pred_mag = pred[:, 2:3]
                    gt_mag = blur_field[:, 2:3]
                    mse = torch.mean((pred_mag - gt_mag) ** 2)
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_psnr += psnr.item()
                    pbar.set_postfix({'loss': loss.item(), 'psnr': psnr.item()})
                    
                    # Save sample visualizations (first batch only)
                    if batch_idx == 0 and epoch % args.save_freq == 0:
                        save_validation_sample(
                            blur_img[0].cpu(),
                            blur_field[0].cpu(),
                            pred[0].cpu(),
                            epoch,
                            args.output_dir
                        )
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        
        # Log validation metrics
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('PSNR/validation', avg_val_psnr, epoch)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        logging.info(f"Epoch {epoch+1}/{args.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Val PSNR: {avg_val_psnr:.2f} dB")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_psnrs': val_psnrs
        }
        
        # Save regular checkpoint
        if epoch % args.save_freq == 0:
            os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth'))
        
        # Check for best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # Save best model
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            logging.info(f"Early stopping counter: {early_stop_counter}/{args.patience}")
            
            if early_stop_counter >= args.patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Plot and save performance curves
        if epoch % args.save_freq == 0:
            plot_performance_curves(train_losses, val_losses, val_psnrs, args.output_dir)
    
    # Final performance plots
    plot_performance_curves(train_losses, val_losses, val_psnrs, args.output_dir)
    
    # Close TensorBoard writer
    writer.close()
    
    logging.info("Training completed!")
    return model

def save_validation_sample(blur_img, gt_field, pred_field, epoch, output_dir):
    """Save validation sample visualizations"""
    # Create directory
    vis_dir = os.path.join(output_dir, 'validation_samples')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save tensors
    gt_path = os.path.join(vis_dir, f'epoch_{epoch}_gt.pt')
    pred_path = os.path.join(vis_dir, f'epoch_{epoch}_pred.pt')
    img_path = os.path.join(vis_dir, f'epoch_{epoch}_blur.png')
    
    # Save blur image
    blur_np = blur_img.permute(1, 2, 0).numpy()
    blur_np = (blur_np * 0.5 + 0.5) * 255  # Denormalize
    blur_np = blur_np.astype(np.uint8)
    plt.imsave(img_path, blur_np)
    
    # Save tensors
    torch.save(gt_field, gt_path)
    torch.save(pred_field, pred_path)
    
    # Visualize blur fields
    gt_vis_path = os.path.join(vis_dir, f'epoch_{epoch}_gt_vis.png')
    pred_vis_path = os.path.join(vis_dir, f'epoch_{epoch}_pred_vis.png')
    
    # Use the visualization function from DPT_blur
    visualize_blur_field_with_legend(
        gt_path, 
        img_path, 
        gt_vis_path, 
        title="Ground Truth Blur Field"
    )
    
    visualize_blur_field_with_legend(
        pred_path, 
        img_path, 
        pred_vis_path, 
        title="Predicted Blur Field"
    )

def plot_performance_curves(train_losses, val_losses, val_psnrs, output_dir):
    """Plot and save performance curves"""
    # Create directory
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'))
    plt.close()
    
    # Plot validation PSNR
    plt.figure(figsize=(10, 5))
    plt.plot(val_psnrs, label='Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'psnr_curve.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MIMO-UNet for blur field prediction')
    
    # Dataset parameters
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation dataset')
    parser.add_argument('--crop_size', type=int, default=256, help='Training crop size')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels in MIMO-UNet')
    parser.add_argument('--model_name', type=str, default='MIMO-UNetPlus', 
                        choices=['MIMO-UNet', 'MIMO-UNetPlus'], 
                        help='Model variant to use')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    # Loss parameters
    parser.add_argument('--lambda_dir', type=float, default=0.5, help='Weight for directional loss')
    parser.add_argument('--lambda_mag', type=float, default=0.5, help='Weight for magnitude loss')
    
    # Checkpoint parameters
    parser.add_argument('--output_dir', type=str, default='blur_field_output', help='Output directory')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint save frequency (epochs)')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Add max_train_samples argument
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use')
    
    args = parser.parse_args()
    
    # Train model
    model = train_model(args)
