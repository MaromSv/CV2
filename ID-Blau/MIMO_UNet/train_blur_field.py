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
import random
import time
from PIL import Image

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

def visualize_dataset_samples(dataset, output_dir, split, num_samples=5):
    """
    Visualize random samples from a dataset
    
    Args:
        dataset: Dataset to sample from
        output_dir: Directory to save visualizations
        split: String identifier (e.g., 'train', 'val')
        num_samples: Number of samples to visualize
    """
    # Create directory
    vis_dir = os.path.join(output_dir, f'{split}_samples')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    print(f"Visualizing {len(indices)} samples from {split} dataset")
    
    # Define fixed color scales for components
    dx_min, dx_max = -0.5, 0.5  # Adjust these based on your typical dx range
    dy_min, dy_max = -0.5, 0.5  # Adjust these based on your typical dy range
    mag_min, mag_max = 0.0, 1.0  # Adjust these based on your typical magnitude range
    
    # Visualize each sample
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Handle different dataset formats
        if isinstance(sample, dict):
            # Dictionary format
            blur_img = sample['blur']
            blur_field = sample['blur_field']
            print(f"{split} sample {i} - Dict format with keys: {sample.keys()}")
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            # List/tuple format
            blur_img = sample[0]
            blur_field = sample[1]
            print(f"{split} sample {i} - List/tuple format with {len(sample)} elements")
        else:
            print(f"Unexpected sample format: {type(sample)}")
            continue
        
        # # Debug print for blur field
        # if isinstance(blur_field, torch.Tensor):
        #     print(f"{split} sample {i} - Blur field shape: {blur_field.shape}")
        #     print(f"{split} sample {i} - Blur field channels: {blur_field.shape[0]}")
        #     print(f"{split} sample {i} - Blur field stats - min: {blur_field.min().item():.4f}, max: {blur_field.max().item():.4f}")
            
        #     # Print stats for each channel
        #     for c in range(blur_field.shape[0]):
        #         channel_name = ["dx", "dy", "magnitude"][c] if blur_field.shape[0] == 3 else f"channel_{c}"
        #         channel_min = blur_field[c].min().item()
        #         channel_max = blur_field[c].max().item()
        #         channel_mean = blur_field[c].mean().item()
        #         print(f"  {split} sample {i} - {channel_name}: min={channel_min:.4f}, max={channel_max:.4f}, mean={channel_mean:.4f}")
                
        #         # Check if channel has uniform values
        #         if abs(channel_max - channel_min) < 1e-5:
        #             print(f"  WARNING: {channel_name} has uniform values! This will result in a single color visualization.")
            
        #     # Check if all values are zero
        #     if torch.all(blur_field == 0):
        #         print(f"  WARNING: Blur field is all zeros! This will result in a black visualization.")
                
        #     # Check if values are very small
        #     if blur_field.abs().max().item() < 0.01:
        #         print(f"  WARNING: Blur field values are very small (max abs: {blur_field.abs().max().item():.6f})! This may result in a very dark visualization.")
        
        # Save blur field tensor
        tensor_path = os.path.join(vis_dir, f'sample_{i}_blur_field.pt')
        torch.save(blur_field, tensor_path)
        
        # Also save as numpy for easier inspection
        np_path = os.path.join(vis_dir, f'sample_{i}_blur_field.npy')
        if isinstance(blur_field, torch.Tensor):
            np.save(np_path, blur_field.cpu().numpy())
        
        # Convert tensor to numpy for visualization
        if isinstance(blur_img, torch.Tensor):
            # Check normalization range by examining min/max values
            min_val = blur_img.min().item()
            max_val = blur_img.max().item()
            print(f"{split} sample {i} - Image tensor range: min={min_val:.4f}, max={max_val:.4f}")
            
            # Denormalize based on detected range
            img_np = blur_img.permute(1, 2, 0).cpu().numpy()
            
            if min_val < -0.4 and max_val < 0.6:
                # Likely normalized to [-0.5, 0.5] range (common in your codebase)
                img_np = img_np + 0.5
                print(f"{split} sample {i} - Denormalizing from [-0.5, 0.5] to [0, 1] range")
            elif min_val < -0.9 and max_val < 1.1:
                # Normalized to [-1, 1] range
                img_np = (img_np + 1) / 2
                print(f"{split} sample {i} - Denormalizing from [-1, 1] to [0, 1] range")
            
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = blur_img
        
        # Debug print to check image values after denormalization
        # print(f"{split} sample {i} - After denormalization - min: {img_np.min():.4f}, max: {img_np.max():.4f}")
        
        # Save image
        img_path = os.path.join(vis_dir, f'sample_{i}_blur.png')
        
        # Ensure image is in the right format for PIL
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        Image.fromarray(img_np).save(img_path)
        
        # Visualize blur field
        try:
            from DPT_blur.visualize_blur_map import visualize_blur_field_with_legend
            vis_path = os.path.join(vis_dir, f'sample_{i}_blur_field.png')
            
            # Add more debugging for visualization
            # print(f"Visualizing blur field for {split} sample {i}")
            # print(f"  Tensor path: {tensor_path}")
            # print(f"  Image path: {img_path}")
            # print(f"  Output path: {vis_path}")
            
            # Try to manually load the tensor to verify it's correct
            loaded_tensor = torch.load(tensor_path)
            # print(f"  Loaded tensor shape: {loaded_tensor.shape}")
            # print(f"  Loaded tensor range: {loaded_tensor.min().item():.4f} to {loaded_tensor.max().item():.4f}")
            
            # Also try to visualize components separately
            components_path = os.path.join(vis_dir, f'sample_{i}_components.png')
            
            # Create a simple component visualization
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot dx with fixed scale
            if loaded_tensor.shape[0] >= 1:
                dx = loaded_tensor[0].cpu().numpy()
                im0 = axs[0].imshow(dx, cmap='coolwarm', vmin=dx_min, vmax=dx_max)
                axs[0].set_title(f'dx (min={dx.min():.4f}, max={dx.max():.4f})')
                plt.colorbar(im0, ax=axs[0])
            
            # Plot dy with fixed scale
            if loaded_tensor.shape[0] >= 2:
                dy = loaded_tensor[1].cpu().numpy()
                im1 = axs[1].imshow(dy, cmap='coolwarm', vmin=dy_min, vmax=dy_max)
                axs[1].set_title(f'dy (min={dy.min():.4f}, max={dy.max():.4f})')
                plt.colorbar(im1, ax=axs[1])
            
            # Plot magnitude with fixed scale
            if loaded_tensor.shape[0] >= 3:
                mag = loaded_tensor[2].cpu().numpy()
                im2 = axs[2].imshow(mag, cmap='viridis', vmin=mag_min, vmax=mag_max)
                axs[2].set_title(f'magnitude (min={mag.min():.4f}, max={mag.max():.4f})')
                plt.colorbar(im2, ax=axs[2])
            
            plt.tight_layout()
            plt.savefig(components_path)
            plt.close()
            
            print(f"  Saved component visualization to {components_path}")
            
            # Now call the actual visualization function
            visualize_blur_field_with_legend(tensor_path, img_path, output_path=vis_path, 
                                           title=f"{split.capitalize()} Sample {i}")
            print(f"Saved {split} blur field visualization to {vis_path}")
        except Exception as e:
            print(f"Error visualizing {split} blur field: {e}")
            import traceback
            traceback.print_exc()

def save_validation_grid(model, fixed_val_loader, epoch, output_dir, device):
    """
    Save a grid visualization of model predictions for fixed validation samples
    
    Args:
        model: The trained model
        fixed_val_loader: Dataloader with fixed validation samples
        epoch: Current epoch number
        output_dir: Directory to save visualizations
        device: Device to run inference on
    """
    # Create directories
    vis_dir = os.path.join(output_dir, 'validation_samples')
    epoch_dir = os.path.join(vis_dir, f'epoch_{epoch}')
    gt_dir = os.path.join(vis_dir, 'gt')
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(epoch_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    print(f"Saving validation grid for epoch {epoch}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store tensors and image paths
    pred_tensors = []
    fixed_images = []
    
    # Process each fixed validation sample
    with torch.no_grad():
        for i, sample in enumerate(fixed_val_loader):
            # Handle different dataset formats
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
            
            print(f"Sample {i} - Input image range: {blur_img.min().item():.4f} to {blur_img.max().item():.4f}")
            print(f"Sample {i} - GT blur field range: {blur_field.min().item():.4f} to {blur_field.max().item():.4f}")
                
            # Save original image
            img_np = blur_img[0].detach().cpu().permute(1, 2, 0).numpy()
            
            # Check if image is normalized and denormalize if needed
            if img_np.min() < -0.1:  # Likely normalized to [-0.5, 0.5]
                img_np = img_np + 0.5
                print(f"Denormalizing image from [-0.5, 0.5] to [0, 1]")
            
            img_np = np.clip(img_np, 0, 1)
            img_path = os.path.join(epoch_dir, f'sample_{i}_blur.png')
            Image.fromarray((img_np * 255).astype(np.uint8)).save(img_path)
            fixed_images.append(img_path)
            
            # Save ground truth to gt directory (only once)
            if not os.path.exists(os.path.join(gt_dir, f'sample_{i}_blur.png')):
                gt_img_path = os.path.join(gt_dir, f'sample_{i}_blur.png')
                Image.fromarray((img_np * 255).astype(np.uint8)).save(gt_img_path)
                
                # Save ground truth blur field tensor
                gt_tensor_path = os.path.join(gt_dir, f'sample_{i}_gt.pt')
                torch.save(blur_field[0].cpu(), gt_tensor_path)
            
            # Forward pass
            outputs = model(blur_img)
            
            # Handle multi-scale outputs if present
            if isinstance(outputs, list):
                # Use the highest resolution output
                pred = outputs[-1]
                print(f"Model returned multi-scale outputs, using highest resolution (shape: {pred.shape})")
            else:
                pred = outputs
                print(f"Model returned single output (shape: {pred.shape})")
            
            # Print prediction stats
            print(f"Prediction range: {pred.min().item():.4f} to {pred.max().item():.4f}")
                
            # Save prediction tensor
            pred_tensor_path = os.path.join(epoch_dir, f'sample_{i}_pred.pt')
            torch.save(pred[0].cpu(), pred_tensor_path)
            
            # Add to list for grid visualization
            pred_tensors.append(pred[0].cpu())
            
            # Also save ground truth tensor in epoch directory for comparison
            gt_tensor_path = os.path.join(epoch_dir, f'sample_{i}_gt.pt')
            torch.save(blur_field[0].cpu(), gt_tensor_path)
    
    # Create grid visualization for predictions
    try:
        from DPT_blur.visualize_blur_map import visualize_multiple_blur_fields
        
        # Create prediction grid
        pred_grid_path = os.path.join(epoch_dir, f'predictions_grid.png')
        visualize_multiple_blur_fields(pred_tensors, fixed_images, pred_grid_path)
        print(f"Saved prediction grid to {pred_grid_path}")
        
        # Create ground truth grid (only once)
        if not os.path.exists(os.path.join(gt_dir, 'gt_grid.png')):
            gt_tensors = [torch.load(os.path.join(gt_dir, f'sample_{i}_gt.pt')) for i in range(len(pred_tensors))]
            gt_images = [os.path.join(gt_dir, f'sample_{i}_blur.png') for i in range(len(pred_tensors))]
            gt_grid_path = os.path.join(gt_dir, 'gt_grid.png')
            visualize_multiple_blur_fields(gt_tensors, gt_images, gt_grid_path)
            print(f"Saved ground truth grid to {gt_grid_path}")
    except Exception as e:
        print(f"Error creating grid visualizations: {e}")
    
    # Set model back to training mode
    model.train()
    
    return os.path.join(epoch_dir, f'predictions_grid.png')

def analyze_dataset_statistics(dataset, name="dataset"):
    """
    Analyze and log statistics about a dataset's blur field distribution
    
    Args:
        dataset: Dataset to analyze
        name: Name identifier for logging
    """
    # Initialize arrays to store statistics
    bx_values = []
    by_values = []
    mag_values = []
    
    # Sample up to 100 random items to avoid memory issues with large datasets
    indices = random.sample(range(len(dataset)), min(100, len(dataset)))
    
    for idx in indices:
        sample = dataset[idx]
        
        # Handle different dataset formats
        if isinstance(sample, dict):
            blur_field = sample['blur_field']
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            blur_field = sample[1]
        else:
            continue
            
        # Extract channels
        if isinstance(blur_field, torch.Tensor):
            bx = blur_field[0].cpu().numpy().flatten()
            by = blur_field[1].cpu().numpy().flatten()
            mag = blur_field[2].cpu().numpy().flatten()
            
            # Sample a subset of pixels to avoid memory issues
            if len(bx) > 1000:
                pixel_indices = random.sample(range(len(bx)), 1000)
                bx = bx[pixel_indices]
                by = by[pixel_indices]
                mag = mag[pixel_indices]
                
            bx_values.extend(bx)
            by_values.extend(by)
            mag_values.extend(mag)
    
    # Calculate statistics
    stats = {
        "bx": {
            "min": np.min(bx_values),
            "max": np.max(bx_values),
            "mean": np.mean(bx_values),
            "std": np.std(bx_values)
        },
        "by": {
            "min": np.min(by_values),
            "max": np.max(by_values),
            "mean": np.mean(by_values),
            "std": np.std(by_values)
        },
        "magnitude": {
            "min": np.min(mag_values),
            "max": np.max(mag_values),
            "mean": np.mean(mag_values),
            "std": np.std(mag_values)
        }
    }
    
    return stats

def train_model(args):
    """Main training function."""
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
    from torchvision import transforms
    import torch.nn.functional as F
    
    # Define transforms for MIMO-UNet (consistent with ID-Blau)
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
        is_train=False,
    )

    # Limit dataset size if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_dataset):
        logging.info(f"Limiting training dataset to {args.max_train_samples} samples (from {len(train_dataset)})")
        # Create a subset of the dataset
        from torch.utils.data import Subset
        indices = list(range(args.max_train_samples))
        train_dataset = Subset(train_dataset, indices)
    
    # Log dataset sizes
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
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
    val_sample_seed = int(time.time()) % 10000  # Use current time as seed
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
    
    # Create model
    logging.info(f"Creating {args.model_name} model...")
    model = build_MIMOUnet_net(
        model_name=args.model_name
    )
    
    # The model already has output layers with 3 channels
    # No need to modify the architecture
    
    model.to(device)
    logging.info(f"Model created: {args.model_name}")
    
    # Multi-scale supervision: build a BlurFieldLoss as our base, then wrap it
    base_loss = BlurFieldLoss(
        lambda_dir=args.lambda_dir,
        lambda_mag=args.lambda_mag
    )
    criterion = MultiScaleBlurFieldLoss(
        base_criterion=base_loss,
        scale_weights=[0.25, 0.5, 1.0],     # head-wise weights: low, mid, full
        use_consistency=True,               # optional up/down consistency
        consistency_weight=0.1
    )
    logging.info("Using multi-scale blur field loss (weights [0.25,0.5,1])")
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=args.patience
    )
    
    # Add a custom print function to log learning rate changes
    current_lr = optimizer.param_groups[0]['lr']
    logging.info(f"Initial learning rate: {current_lr}")
    
    # Training loop
    logging.info("Starting training...")
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Lists to store metrics for plotting
    epochs_list = []
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_mses = []  # Add MSE tracking list
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different dataset formats
            if isinstance(batch, dict):
                # Dictionary format
                blur_img = batch['blur'].to(device)
                blur_field = batch['blur_field'].to(device)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # List/tuple format
                blur_img = batch[0].to(device)
                blur_field = batch[1].to(device)
            else:
                logging.warning(f"Unexpected batch format: {type(batch)}")
                continue
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(blur_img)
            
            # Compute loss
            loss, losses_dict = criterion(outputs, blur_field)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Log losses
            writer.add_scalar('Loss/total', losses_dict['loss_total'], epoch * len(train_loader) + batch_idx)

            # Log the per-scale and consistency components
            writer.add_scalar('Loss/low',       losses_dict['loss_low'],       epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/medium',    losses_dict['loss_medium'],    epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/high',      losses_dict['loss_high'],      epoch * len(train_loader) + batch_idx)
            if 'loss_consistency' in losses_dict:
                writer.add_scalar('Loss/consistency', losses_dict['loss_consistency'], epoch * len(train_loader) + batch_idx)
            
            # Update metrics
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_mse = 0.0  # Add MSE tracking
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle different dataset formats
                if isinstance(batch, dict):
                    # Dictionary format
                    blur_img = batch['blur'].to(device)
                    blur_field = batch['blur_field'].to(device)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    # List/tuple format
                    blur_img = batch[0].to(device)
                    blur_field = batch[1].to(device)
                else:
                    logging.warning(f"Unexpected batch format: {type(batch)}")
                    continue
                
                # Forward pass
                outputs = model(blur_img)
                
                # Compute loss
                loss, losses_dict = criterion(outputs, blur_field)
                
                # Update metrics
                val_loss += loss.item()
                
                # Calculate MSE and PSNR (for monitoring only)
                if isinstance(outputs, list):
                    pred = outputs[-1]
                else:
                    pred = outputs
                
                mse = torch.mean((pred - blur_field) ** 2)
                val_mse += mse.item()  # Track MSE
                val_psnr += 10 * torch.log10(1.0 / mse).item()
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_mse /= len(val_loader)  # Average MSE
        
        # Store metrics for plotting
        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_mses.append(val_mse)  # Store MSE for plotting
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('PSNR/val', val_psnr, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)  # Log MSE to TensorBoard
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # Log if learning rate changed
        if new_lr != old_lr:
            logging.info(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }
            os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint at regular intervals
        if epoch % args.save_freq == 0:
            os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth'))
            
            # Save validation grid visualization
            logging.info(f"Creating validation grid visualization for epoch {epoch}...")
            grid_path = save_validation_grid(model, fixed_val_loader, epoch, args.output_dir, device)
            logging.info(f"Saved validation grid to {grid_path}")
    
    # Final validation grid visualization
    logging.info(f"Creating final validation grid visualization...")
    grid_path = save_validation_grid(model, fixed_val_loader, args.epochs, args.output_dir, device)
    logging.info(f"Saved final validation grid to {grid_path}")
    
    # Load best model for final evaluation
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logging.info(f"Loaded best model from epoch {best_checkpoint['epoch']+1} with validation loss: {best_checkpoint['best_val_loss']:.4f}")
    
    # Final validation grid visualization with best model
    logging.info(f"Creating validation grid visualization with best model...")
    grid_path = save_validation_grid(model, fixed_val_loader, 'best', args.output_dir, device)
    logging.info(f"Saved best model validation grid to {grid_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Generate and save plots
    logging.info("Generating training metrics plots...")
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'loss_plot.png'), dpi=300)
    plt.close()
    
    # Plot PSNR
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, val_psnrs, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'psnr_plot.png'), dpi=300)
    plt.close()
    
    # Plot MSE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, val_mses, 'm-')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Validation MSE')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'mse_plot.png'), dpi=300)
    plt.close()
    
    # Combined plot with all metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_list, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_list, val_psnrs, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_list, val_mses, 'm-')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Validation MSE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_metrics.png'), dpi=300)
    plt.close()
    
    logging.info(f"Training metrics plots saved to {plots_dir}")
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train MIMO-UNet for blur field prediction')
    
    # Dataset arguments
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size for training')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='MIMO-UNetPlus', help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=64, help='Number of base channels in the model')
    parser.add_argument('--num_scales', type=int, default=3, help='Number of scales in the model')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of saving checkpoints')
    parser.add_argument('--patience', type=int, default=10, help='Patience for learning rate scheduler')
    
    # Loss arguments
    parser.add_argument('--multi_scale_loss', action='store_true', help='Use multi-scale loss')
    parser.add_argument('--lambda_dir', type=float, default=1.0, help='Weight for direction loss')
    parser.add_argument('--lambda_mag', type=float, default=1.0, help='Weight for magnitude loss')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
