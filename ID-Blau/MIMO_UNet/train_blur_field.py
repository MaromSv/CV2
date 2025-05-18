import os
import sys
import copy
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
from blur_losses import MultiScaleBlurFieldLoss, BlurFieldLoss

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
        blurred_dir=os.path.join(args.train_dir, 'blur'),
        gt_dir=os.path.join(args.train_dir, 'condition'),
        transform=None,  # Add transforms if needed
        crop_size=args.crop_size,
        # is_train=True
        is_train=False
    )
    
    # Limit dataset size if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_dataset):
        logging.info(f"Limiting training dataset to {args.max_train_samples} samples (from {len(train_dataset)})")
        # Create a subset of the dataset
        from torch.utils.data import Subset
        indices = list(range(args.max_train_samples))
        train_dataset = Subset(train_dataset, indices)
    
    # val_dataset = BlurMapDataset(
    #     blurred_dir=os.path.join(args.val_dir, 'blur'),
    #     gt_dir=os.path.join(args.val_dir, 'condition'),
    #     transform=None,
    #     crop_size=args.crop_size,
    #     is_train=False
    # )
    val_dataset = copy.deepcopy(train_dataset)
    
    # Visualize dataset samples
    logging.info("Visualizing dataset samples...")
    visualize_dataset_samples(train_dataset, args.output_dir, "train", num_samples=10)
    visualize_dataset_samples(val_dataset, args.output_dir, "val", num_samples=10)
    
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
    if args.use_multi_scale_loss:
        # Create base criterion
        base_criterion = BlurFieldLoss(lambda_dir=0.5, lambda_mag=0.5)
        
        # Create multi-scale loss
        criterion = MultiScaleBlurFieldLoss(
            base_criterion=base_criterion,
            use_consistency=True,
            consistency_weight=args.consistency_weight
        )
        logging.info("Using multi-scale loss with consistency")
    else:
        # Use standard loss
        criterion = BlurFieldLoss(lambda_dir=0.5, lambda_mag=0.5)
        logging.info("Using standard blur field loss")
    
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
                # Check the type of sample and handle accordingly
                if isinstance(sample, dict):
                    # Dictionary format
                    blur_img = sample['blur'].to(device)
                    blur_field = sample['blur_field'].to(device)
                elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    # List/tuple format
                    blur_img = sample[0].to(device)
                    blur_field = sample[1].to(device)
                else:
                    print(f"Unexpected sample format: {type(sample)}")
                    continue
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(blur_img)
                
                # Compute loss based on output type
                if isinstance(outputs, list) and args.use_multi_scale_loss:
                    # For multi-scale outputs with multi-scale loss
                    
                    # Get adaptive scale weights based on training progress
                    if args.adaptive_weights:
                        progress = epoch / args.epochs
                        if progress < 0.2:
                            scale_weights = [0.5, 0.3, 0.2]  # Early: focus on low resolution
                        elif progress < 0.5:
                            scale_weights = [0.3, 0.4, 0.3]  # Mid: balanced focus
                        elif progress < 0.8:
                            scale_weights = [0.2, 0.3, 0.5]  # Later: focus on high resolution
                        else:
                            scale_weights = [0.1, 0.3, 0.6]  # Final: heavy focus on high resolution
                    
                    # Update loss function weights
                    criterion.update_scale_weights(scale_weights)
                    
                    # Compute multi-scale loss
                    loss, losses_dict = criterion(outputs, blur_field)
                    
                    # Log individual losses
                    if writer is not None:
                        for loss_name, loss_value in losses_dict.items():
                            writer.add_scalar(f'Loss/{loss_name}', loss_value, 
                                             epoch * len(train_loader) + batch_idx)
                elif isinstance(outputs, list):
                    # For multi-scale outputs with standard loss (use only highest resolution)
                    pred = outputs[-1]
                    loss = criterion(pred, blur_field)
                else:
                    # For single output
                    loss = criterion(outputs, blur_field)
                
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
                    # Check the type of sample and handle accordingly
                    if isinstance(sample, dict):
                        # Dictionary format
                        blur_img = sample['blur'].to(device)
                        blur_field = sample['blur_field'].to(device)
                    elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                        # List/tuple format
                        blur_img = sample[0].to(device)
                        blur_field = sample[1].to(device)
                    else:
                        print(f"Unexpected sample format: {type(sample)}")
                        continue
                    
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

def save_validation_sample(blur_img, gt_field, pred_field, epoch, output_dir, val_loader=None):
    """Save validation sample visualizations with color wheel legends"""
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from PIL import Image
    import sys
    
    # Add parent directory to path to ensure imports work
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # Import the visualization function from DPT_blur
    try:
        from DPT_blur.visualize_blur_map import create_color_wheel, visualize_blur_field_with_legend
        print("Successfully imported visualization functions")
    except ImportError as e:
        print(f"Error importing visualization functions: {e}")
        return None, None
    
    # Create directory
    vis_dir = os.path.join(output_dir, 'validation_samples')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create epoch-specific directory
    epoch_dir = os.path.join(vis_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Save the blur image (properly denormalized)
    img_path = os.path.join(epoch_dir, f'sample_0_blur.png')
    
    # Convert tensor to numpy and move channels to last dimension
    blur_np = blur_img.detach().cpu().permute(1, 2, 0).numpy()
    
    # Print value range to diagnose normalization
    min_val, max_val = blur_np.min(), blur_np.max()
    print(f"Validation blur image range: min={min_val:.4f}, max={max_val:.4f}")
    
    # Apply appropriate denormalization based on the value range
    if min_val < -0.9 and max_val > 0.9:
        # Likely in [-1, 1] range (from NormalizeImage with mean=0.5, std=0.5)
        blur_np = blur_np * 0.5 + 0.5
    elif min_val < -0.4 and max_val > 0.4:
        # Likely in [-0.5, 0.5] range
        blur_np = blur_np + 0.5
    elif min_val < 0:
        # Some other negative range normalization
        blur_np = (blur_np - min_val) / (max_val - min_val)
    
    # Ensure values are in valid range
    blur_np = np.clip(blur_np, 0, 1)
    
    # Save using PIL for better quality
    Image.fromarray((blur_np * 255).astype(np.uint8)).save(img_path)
    
    # Create a fixed set of validation samples directory
    fixed_samples_dir = os.path.join(vis_dir, 'fixed_samples')
    os.makedirs(fixed_samples_dir, exist_ok=True)
    
    # Save tensors for the current batch
    gt_path = os.path.join(fixed_samples_dir, f'sample_0_gt.pt')
    pred_path = os.path.join(fixed_samples_dir, f'sample_0_pred.pt')
    
    # Save tensors for blur field visualization
    torch.save(gt_field, gt_path)
    torch.save(pred_field, pred_path)
    
    # Create 5 samples (duplicate the current one for simplicity)
    for i in range(5):
        fixed_gt_path = os.path.join(fixed_samples_dir, f'sample_{i}_gt.pt')
        fixed_pred_path = os.path.join(fixed_samples_dir, f'sample_{i}_pred.pt')
        fixed_img_path = os.path.join(fixed_samples_dir, f'sample_{i}_blur.png')
        
        # Save the ground truth tensor
        torch.save(gt_field, fixed_gt_path)
        
        # Save the prediction tensor
        torch.save(pred_field, fixed_pred_path)
        
        # Save the blur image
        Image.fromarray((blur_np * 255).astype(np.uint8)).save(fixed_img_path)
    
    # Create a grid of GT blur field visualizations with color wheels
    gt_grid_path = os.path.join(vis_dir, f'epoch_{epoch}_gt_grid.png')
    
    # Create a figure with 5 subplots for ground truth
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'Ground Truth Blur Fields - Epoch {epoch}', fontsize=16)
    
    # Create color wheel for legend (once)
    wheel = create_color_wheel(size=200)
    
    # Add color wheel to the first subplot
    wheel_ax = fig.add_axes([0.01, 0.15, 0.15, 0.7])  # [left, bottom, width, height]
    wheel_ax.imshow(wheel)
    wheel_ax.set_title("Color Legend", fontsize=12)
    wheel_ax.axis('off')
    
    # Add orientation labels to the color wheel
    radius = 110
    center = 100
    wheel_ax.text(center, center-radius-10, "90°", ha='center', va='center', fontweight='bold', color='black')
    wheel_ax.text(center+radius+10, center, "0°", ha='center', va='center', fontweight='bold', color='black')
    wheel_ax.text(center, center+radius+10, "270°", ha='center', va='center', fontweight='bold', color='black')
    wheel_ax.text(center-radius-10, center, "180°", ha='center', va='center', fontweight='bold', color='black')
    
    # Process and display each GT sample
    for i in range(5):
        # Load the ground truth tensor
        gt_tensor = torch.load(os.path.join(fixed_samples_dir, f'sample_{i}_gt.pt'))
        
        # Convert to numpy
        gt_np = gt_tensor.detach().cpu().numpy()
        
        # Extract components
        bx = gt_np[0]
        by = gt_np[1]
        magnitude = gt_np[2]
        
        # Calculate orientation
        orientation = np.arctan2(by, bx)
        
        # Create HSV representation
        hue = (orientation + np.pi) / (2 * np.pi)
        saturation = np.clip(magnitude / magnitude.max() if magnitude.max() > 0 else magnitude, 0, 1)
        value = np.ones_like(magnitude)
        
        # Stack HSV channels
        hsv = np.stack([hue, saturation, value], axis=-1)
        
        # Convert to RGB
        rgb = mcolors.hsv_to_rgb(hsv)
        
        # Display
        axes[i].imshow(rgb)
        axes[i].set_title(f'GT Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0.15, 0, 1, 0.95])  # Adjust layout to make room for the color wheel
    plt.savefig(gt_grid_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Create a grid of predicted blur field visualizations with color wheels
    pred_grid_path = os.path.join(vis_dir, f'epoch_{epoch}_pred_grid.png')
    
    # Create a figure with 5 subplots for predictions
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'Predicted Blur Fields - Epoch {epoch}', fontsize=16)
    
    # Add color wheel to the first subplot
    wheel_ax = fig.add_axes([0.01, 0.15, 0.15, 0.7])  # [left, bottom, width, height]
    wheel_ax.imshow(wheel)
    wheel_ax.set_title("Color Legend", fontsize=12)
    wheel_ax.axis('off')
    
    # Add orientation labels to the color wheel
    wheel_ax.text(center, center-radius-10, "90°", ha='center', va='center', fontweight='bold', color='black')
    wheel_ax.text(center+radius+10, center, "0°", ha='center', va='center', fontweight='bold', color='black')
    wheel_ax.text(center, center+radius+10, "270°", ha='center', va='center', fontweight='bold', color='black')
    wheel_ax.text(center-radius-10, center, "180°", ha='center', va='center', fontweight='bold', color='black')
    
    # Process and display each prediction sample
    for i in range(5):
        # Load the prediction tensor
        pred_tensor = torch.load(os.path.join(fixed_samples_dir, f'sample_{i}_pred.pt'))
        
        # Convert to numpy
        pred_np = pred_tensor.detach().cpu().numpy()
        
        # Extract components
        bx = pred_np[0]
        by = pred_np[1]
        magnitude = pred_np[2]
        
        # Calculate orientation
        orientation = np.arctan2(by, bx)
        
        # Create HSV representation
        hue = (orientation + np.pi) / (2 * np.pi)
        saturation = np.clip(magnitude / magnitude.max() if magnitude.max() > 0 else magnitude, 0, 1)
        value = np.ones_like(magnitude)
        
        # Stack HSV channels
        hsv = np.stack([hue, saturation, value], axis=-1)
        
        # Convert to RGB
        rgb = mcolors.hsv_to_rgb(hsv)
        
        # Display
        axes[i].imshow(rgb)
        axes[i].set_title(f'Pred Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0.15, 0, 1, 0.95])  # Adjust layout to make room for the color wheel
    plt.savefig(pred_grid_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Also create individual visualizations with the specialized function
    # This creates a nicer visualization with the color wheel legend
    gt_vis_path = os.path.join(vis_dir, f'epoch_{epoch}_gt_vis.png')
    pred_vis_path = os.path.join(vis_dir, f'epoch_{epoch}_pred_vis.png')
    
    # Use the specialized visualization function
    try:
        visualize_blur_field_with_legend(
            tensor_path=gt_path,
            image_path=img_path,
            output_path=gt_vis_path,
            title="Ground Truth Blur Field"
        )
        
        visualize_blur_field_with_legend(
            tensor_path=pred_path,
            image_path=img_path,
            output_path=pred_vis_path,
            title="Predicted Blur Field"
        )
    except Exception as e:
        print(f"Error creating specialized visualizations: {e}")
    
    print(f"Saved validation visualizations to {vis_dir}")
    return gt_grid_path, pred_grid_path

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

def visualize_dataset_samples(dataset, output_dir, prefix, num_samples=10):
    """
    Visualize samples from a dataset and save them to disk
    
    Args:
        dataset: PyTorch dataset
        output_dir: Directory to save visualizations
        prefix: Prefix for filenames (e.g., 'train' or 'val')
        num_samples: Number of samples to visualize
    """
    # Create directory
    vis_dir = os.path.join(output_dir, 'dataset_visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a dataloader with batch size 1
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Get samples
    for i, sample in enumerate(loader):
        if i >= num_samples:
            break
            
        # Handle different dataset formats
        if isinstance(sample, dict) and 'blur' in sample:
            # Dictionary format with 'blur' key
            img = sample['blur'][0]  # Remove batch dimension
        elif isinstance(sample, (list, tuple)) and len(sample) > 0:
            # List/tuple format, assume first element is image
            img = sample[0][0]  # Remove batch dimension
        else:
            # Direct tensor format
            img = sample[0]  # Remove batch dimension
            
        # Convert to numpy and adjust range for visualization
        if img.shape[0] == 3:  # RGB image
            # Convert from CxHxW to HxWxC
            img_np = img.permute(1, 2, 0).cpu().numpy()
            
            # Adjust range based on typical normalization
            if img_np.min() < 0:
                # Likely normalized to [-0.5, 0.5] or [-1, 1]
                img_np = (img_np + 0.5) if img_np.min() >= -0.5 else (img_np + 1.0) / 2.0
            elif img_np.max() <= 1.0:
                # Already in [0, 1] range
                pass
            else:
                # Likely in [0, 255] range
                img_np = img_np / 255.0
                
            # Ensure in [0, 1] range
            img_np = np.clip(img_np, 0, 1)
        else:
            # Grayscale or other format - just take first channel
            img_np = img[0].cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Save image
        plt.figure(figsize=(8, 8))
        plt.imshow(img_np)
        plt.title(f"{prefix} Sample {i+1}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{prefix}_sample_{i+1}.png"))
        plt.close()
        
        # Also save the raw tensor for reference
        torch.save(img, os.path.join(vis_dir, f"{prefix}_sample_{i+1}.pt"))
        
        # If we have blur field data, visualize it too
        if isinstance(sample, dict) and 'blur_field' in sample:
            blur_field = sample['blur_field'][0]  # Remove batch dimension
            
            # Save tensor
            torch.save(blur_field, os.path.join(vis_dir, f"{prefix}_blur_field_{i+1}.pt"))
            
            # Try to visualize if it has the expected format (3 channels)
            if blur_field.shape[0] == 3:
                # Extract components (assuming bx, by, magnitude format)
                bx = blur_field[0].cpu().numpy()
                by = blur_field[1].cpu().numpy()
                magnitude = blur_field[2].cpu().numpy()
                
                # Create visualization
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(bx, cmap='coolwarm')
                plt.title('X Direction')
                plt.colorbar()
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(by, cmap='coolwarm')
                plt.title('Y Direction')
                plt.colorbar()
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(magnitude, cmap='viridis')
                plt.title('Magnitude')
                plt.colorbar()
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"{prefix}_blur_field_viz_{i+1}.png"))
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
    
    # Add arguments for multi-scale loss
    parser.add_argument('--use_multi_scale_loss', action='store_true', 
                        help='Use multi-scale loss for training')
    parser.add_argument('--consistency_weight', type=float, default=0.1,
                        help='Weight for consistency loss between scales')
    
    args = parser.parse_args()
    
    # Train model
    model = train_model(args)
