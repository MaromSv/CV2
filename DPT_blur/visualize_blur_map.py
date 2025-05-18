import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2 # Import OpenCV

def visualize_blur_map(tensor_path, image_path=None, quiver_step=16, output_path=None):
    """Loads a saved blur vector (bx, by) tensor and visualizes its calculated magnitude, orientation, and the vector field itself, optionally superimposed on the original image."""

    if not os.path.exists(tensor_path):
        print(f"Error: File not found at {tensor_path}")
        return

    original_image = None
    if image_path:
        if not os.path.exists(image_path):
            print(f"Warning: Original image file not found at {image_path}. Proceeding without it.")
            image_path = None # Reset to proceed without image
        else:
            try:
                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"Warning: Failed to load original image from {image_path}. Proceeding without it.")
                    image_path = None
                else:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    print(f"Loaded original image from {image_path} with shape: {original_image.shape}")
            except Exception as e:
                print(f"Error loading original image {image_path}: {e}. Proceeding without it.")
                image_path = None
                original_image = None

    try:
        img_basename = os.path.basename(original_image_path_from_pt_file)
        img_name_no_ext = os.path.splitext(img_basename)[0]
        gt_npy_filename = img_name_no_ext + ".npy"

        blur_dir = os.path.dirname(original_image_path_from_pt_file)
        split_dir = os.path.dirname(blur_dir) # e.g., .../train/ or .../val/
        if os.path.basename(blur_dir) != 'blur':
            print(f"Warning: Expected 'blur' in path {original_image_path_from_pt_file} for GT derivation, found {os.path.basename(blur_dir)}.")
            return None 
            
        gt_condition_dir = os.path.join(split_dir, "condition")
        gt_full_path = os.path.join(gt_condition_dir, gt_npy_filename)
        return gt_full_path
    except Exception as e:
        print(f"Error deriving GT path from {original_image_path_from_pt_file}: {e}")
        return None

def process_blur_data(blur_data_np, data_label="Data"):
    """Processes a 3-channel numpy array (cos, sin, mag) into components for visualization."""
    cos_comp = blur_data_np[0, :, :]
    sin_comp = blur_data_np[1, :, :]
    mag_map = blur_data_np[2, :, :]
    H, W = cos_comp.shape # Get dimensions from the data itself
    
    orientation_for_hsv = np.arctan2(sin_comp, cos_comp) # For HSV plot
    bx_for_quiver = mag_map * cos_comp # For quiver plot
    by_for_quiver = mag_map * sin_comp # For quiver plot

    print(f"\n--- {data_label} --- ")
    print(f"  Shape: ({H}, {W})") # Print shape
    print(f"  Cos Comp (Ch0) - Min: {cos_comp.min():.4f}, Max: {cos_comp.max():.4f}, Mean: {cos_comp.mean():.4f}")
    print(f"  Sin Comp (Ch1) - Min: {sin_comp.min():.4f}, Max: {sin_comp.max():.4f}, Mean: {sin_comp.mean():.4f}")
    print(f"  Magnitude (Ch2) - Min: {mag_map.min():.4f}, Max: {mag_map.max():.4f}, Mean: {mag_map.mean():.4f}")
    print(f"  bx Quiver (Mag*Cos) - Min: {bx_for_quiver.min():.4f}, Max: {bx_for_quiver.max():.4f}, Mean: {bx_for_quiver.mean():.4f}")
    print(f"  by Quiver (Mag*Sin) - Min: {by_for_quiver.min():.4f}, Max: {by_for_quiver.max():.4f}, Mean: {by_for_quiver.mean():.4f}")
    return cos_comp, sin_comp, mag_map, orientation_for_hsv, bx_for_quiver, by_for_quiver, H, W # Return H, W

def plot_visualization_row(axes_row, cos_comp, sin_comp, mag_map, orientation_hsv, bx_quiver, by_quiver, 
                           original_image_display, quiver_step, data_H, data_W, row_title_prefix=""):
    """Plots one row (3 subplots) for either GT or Prediction."""
    
    # Plot 1: Colorwheel (Orientation-Hue, Magnitude-Saturation)
    hue_channel = (orientation_hsv + np.pi) / (2 * np.pi)
    min_mag_for_sat, max_mag_for_sat = np.min(mag_map), np.max(mag_map)
    if max_mag_for_sat == min_mag_for_sat:
        saturation_channel = np.zeros_like(mag_map)
    else:
        saturation_channel = (mag_map - min_mag_for_sat) / (max_mag_for_sat - min_mag_for_sat)
    saturation_channel = np.clip(saturation_channel, 0, 1)
    value_channel = np.ones_like(mag_map)
    hsv_colorwheel_image = np.stack([hue_channel, saturation_channel, value_channel], axis=-1)
    try:
        import matplotlib.colors
        rgb_colorwheel_image = matplotlib.colors.hsv_to_rgb(hsv_colorwheel_image)
    except ImportError:
        rgb_colorwheel_image = plt.matplotlib.colors.hsv_to_rgb(hsv_colorwheel_image)
    axes_row[0].imshow(rgb_colorwheel_image)
    axes_row[0].set_title(f'{row_title_prefix} HSV (Angle-Hue, Mag-Sat)')
    axes_row[0].set_xticks([]); axes_row[0].set_yticks([])

    # Plot 2: Magnitude Heatmap
    im_mag = axes_row[1].imshow(mag_map, cmap='viridis')
    axes_row[1].set_title(f'{row_title_prefix} Magnitude')
    axes_row[1].set_xticks([]); axes_row[1].set_yticks([])
    plt.gcf().colorbar(im_mag, ax=axes_row[1], fraction=0.046, pad=0.04)

    # Plot 3: Vector Field (Quiver)
    x_coords, y_coords = np.meshgrid(np.arange(0, data_W), np.arange(0, data_H))
    step = max(1, quiver_step)
    X_sub, Y_sub = x_coords[::step, ::step], y_coords[::step, ::step]
    bx_sub, by_sub = bx_quiver[::step, ::step], by_quiver[::step, ::step]
    mag_sub_for_color = mag_map[::step, ::step]

    if original_image_display is not None:
        if original_image_display.shape[0] != data_H or original_image_display.shape[1] != data_W:
            original_image_resized = cv2.resize(original_image_display, (data_W, data_H), interpolation=cv2.INTER_AREA)
            axes_row[2].imshow(original_image_resized)
        else:
            axes_row[2].imshow(original_image_display)
    else:
        axes_row[2].imshow(np.ones((data_H, data_W)) * 0.5, cmap='gray', vmin=0, vmax=1)
    
    quiv = axes_row[2].quiver(X_sub, Y_sub, bx_sub, -by_sub, mag_sub_for_color,
                              cmap='viridis', scale=None, scale_units='xy', 
                              angles='xy', width=0.003, pivot='middle')
    axes_row[2].set_title(f'{row_title_prefix} Vector Field (Step={step})')
    axes_row[2].set_aspect('equal', adjustable='box'); axes_row[2].set_xlim(0, data_W); axes_row[2].set_ylim(data_H, 0)
    axes_row[2].set_xticks([]); axes_row[2].set_yticks([])
    plt.gcf().colorbar(quiv, ax=axes_row[2], fraction=0.046, pad=0.04, label='Vector Magnitude')


def visualize_blur_map(predicted_tensor_path, image_path_cli=None, quiver_step=16):
    if not os.path.exists(predicted_tensor_path):
        print(f"Error: Predicted tensor file not found at {predicted_tensor_path}")
        return

    # --- Load Prediction Data ---
    loaded_pred_image_path = None
    pred_blur_map_tensor = None
    try:
        # Explicitly set weights_only=False as we are loading a dictionary, not just model weights.
        # This is important for PyTorch 2.6+ where the default changed.
        loaded_data = torch.load(predicted_tensor_path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(loaded_data, dict):
            pred_blur_map_tensor = loaded_data['blur_map_tensor']
            loaded_pred_image_path = loaded_data['original_image_path']
        elif isinstance(loaded_data, torch.Tensor):
            pred_blur_map_tensor = loaded_data
        else: raise TypeError("Predicted file is not dict or tensor.")
        pred_blur_map_np = pred_blur_map_tensor.detach().cpu().numpy()
        if pred_blur_map_np.shape[0] != 3: raise ValueError("Prediction tensor needs 3 channels.")
    except Exception as e:
        print(f"Error loading PRED data from {predicted_tensor_path}: {e}"); return

    cos_pred, sin_pred, mag_pred, orientation_hsv_pred, bx_quiver_pred, by_quiver_pred, H_pred, W_pred = \
        process_blur_data(pred_blur_map_np, "Prediction")

    # --- Determine and Load Original Image for Display ---
    final_display_image_path = image_path_cli if image_path_cli else loaded_pred_image_path
    original_image_for_display = None
    if final_display_image_path:
        if os.path.exists(final_display_image_path):
            try:
                original_image_for_display = cv2.cvtColor(cv2.imread(final_display_image_path), cv2.COLOR_BGR2RGB)
            except Exception as e: print(f"Error loading display image {final_display_image_path}: {e}")
        else: print(f"Display image not found: {final_display_image_path}")
    else: print("No path for original image display.")

    # --- Load Ground Truth Data ---
    gt_map_np = None
    gt_path = derive_gt_path(final_display_image_path) # Use final_display_image_path to find GT
    if gt_path and os.path.exists(gt_path):
        try:
            gt_map_np_loaded = np.load(gt_path)
            # Handle potential NaNs in GT for visualization purposes
            gt_map_np = np.nan_to_num(gt_map_np_loaded, nan=0.0, posinf=0.0, neginf=0.0)
            if gt_map_np.shape[0] != 3: raise ValueError("GT tensor needs 3 channels.")
            print(f"Loaded GT from {gt_path}")
        except Exception as e: print(f"Error loading GT from {gt_path}: {e}"); gt_map_np = None
    elif gt_path: print(f"GT file not found at derived path: {gt_path}")
    else: print("Could not derive GT path or no original image path available.")

    # --- Setup Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)) # Adjusted figsize for 2 rows
    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.1)

    # --- Plot Top Row: Ground Truth (if available) ---
    if gt_map_np is not None:
        cos_gt, sin_gt, mag_gt, orientation_hsv_gt, bx_quiver_gt, by_quiver_gt, H_gt, W_gt = \
            process_blur_data(gt_map_np, "Ground Truth")
        plot_visualization_row(axes[0,:], cos_gt, sin_gt, mag_gt, orientation_hsv_gt, 
                               bx_quiver_gt, by_quiver_gt, original_image_for_display, 
                               quiver_step, H_gt, W_gt, "GT")
    else:
        for i in range(3): axes[0,i].text(0.5, 0.5, 'Ground Truth Not Available', ha='center', va='center'); axes[0,i].set_xticks([]); axes[0,i].set_yticks([])
        axes[0,0].set_title("GT HSV (Not Available)")
        axes[0,1].set_title("GT Magnitude (Not Available)")
        axes[0,2].set_title("GT Vector Field (Not Available)")

    # --- Plot Bottom Row: Predictions ---
    plot_visualization_row(axes[1,:], cos_pred, sin_pred, mag_pred, orientation_hsv_pred, 
                           bx_quiver_pred, by_quiver_pred, original_image_for_display, 
                           quiver_step, H_pred, W_pred, "Pred")

    plt.suptitle(f'Blur Vector Visualization ({os.path.basename(tensor_path)})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjusted rect to accommodate suptitle with subplots_adjust
    
    # Save to file if output_path is provided, otherwise show
    if output_path:
        print(f"Saving visualization to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predicted and GT blur maps.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the saved PREDICTED blur data tensor (.pt file from inference).')
    parser.add_argument('--image_path', type=str, default=None, help='Optional path to the original input image. Overrides path in .pt file.')
    parser.add_argument('--step', type=int, default=16, help='Step size for quiver plot subsampling.')
    args = parser.parse_args()

    visualize_blur_map(args.input_file, image_path=args.image_path, quiver_step=args.step) 

def create_color_wheel(size=200, with_labels=True):
    """Create a color wheel to visualize orientation and magnitude mapping."""
    # Create a white background
    wheel = np.ones((size, size, 3))
    
    # Create a grid of coordinates
    y, x = np.ogrid[0:size, 0:size]
    
    # Center coordinates
    center_y, center_x = size // 2, size // 2
    
    # Calculate distance from center and angle
    y = y - center_y
    x = x - center_x
    
    # Calculate radius and angle
    radius = np.sqrt(x*x + y*y)
    angle = np.arctan2(y, x)  # Range: [-pi, pi]
    
    # Normalize radius to [0, 1]
    max_radius = size // 2
    radius_norm = np.clip(radius / max_radius, 0, 1)
    
    # Convert angle to hue (0-1 range)
    hue = (angle + np.pi) / (2 * np.pi)  # Range: [0, 1]
    
    # Create HSV image
    hsv = np.zeros((size, size, 3))
    hsv[:, :, 0] = hue  # Hue = orientation
    hsv[:, :, 1] = np.ones_like(radius)  # Full saturation
    hsv[:, :, 2] = np.ones_like(radius)  # Full value
    
    # Convert HSV to RGB
    import matplotlib.colors as mcolors
    colored_wheel = mcolors.hsv_to_rgb(hsv)
    
    # Create a circular mask
    mask = radius_norm <= 1.0
    
    # Apply mask - colored wheel on white background
    wheel = np.where(mask[:, :, np.newaxis], colored_wheel, wheel)
    
    # Add a black border
    border_mask = (radius_norm > 0.95) & (radius_norm <= 1.0)
    wheel[border_mask] = [1, 1, 1]
    
    return wheel

def visualize_blur_field_with_legend(tensor_path, image_path=None, output_path=None, title="Blur Condition Field"):
    """
    Visualizes a blur field tensor with a color wheel legend showing orientation and magnitude.
    
    Args:
        tensor_path: Path to the saved blur tensor (.pt file)
        image_path: Optional path to the original image
        output_path: Path to save the visualization
        title: Title for the visualization
    """
    if not os.path.exists(tensor_path):
        print(f"Error: File not found at {tensor_path}")
        return
    
    try:
        # Load the tensor
        blur_map_tensor = torch.load(tensor_path, map_location=torch.device('cpu'))
        print(f"Loaded tensor from {tensor_path} with shape: {blur_map_tensor.shape}")
        
        if not isinstance(blur_map_tensor, torch.Tensor):
            raise TypeError("Loaded object is not a torch.Tensor")
        if blur_map_tensor.dim() != 3 or blur_map_tensor.shape[0] != 3:
            raise ValueError(f"Expected tensor shape (3, H, W) for (bx, by, magnitude), but got {blur_map_tensor.shape}")
        
        # Detach and convert to numpy
        blur_map_np = blur_map_tensor.detach().cpu().numpy()
        
    except Exception as e:
        print(f"Error loading or processing tensor from {tensor_path}: {e}")
        return
    
    # Extract components
    bx = blur_map_np[0, :, :]
    by = blur_map_np[1, :, :]
    magnitude_map = blur_map_np[2, :, :]
    H, W = bx.shape
    
    # Calculate orientation (angle) from bx and by
    orientation_map = np.arctan2(by, bx)  # Range: [-pi, pi]
    
    # Create figure with two parts: legend and visualization
    fig = plt.figure(figsize=(12, 5))
    
    # Create a GridSpec to control the layout
    gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
    
    # Add the color wheel legend
    ax_legend = fig.add_subplot(gs[0, 0])
    
    # Create and display the color wheel
    wheel = create_color_wheel(size=200)
    ax_legend.imshow(wheel)
    
    # Add title above the color wheel
    plt.figtext(0.16, 0.88, title, ha='center', fontsize=15, fontweight='bold')
    
    # Add orientation labels
    radius = 110
    ax_legend.text(100, 100-radius-10, "90°", ha='center', va='center', fontweight='bold', color='black')
    ax_legend.text(100+radius+10, 100, "0°", ha='center', va='center', fontweight='bold', color='black')
    ax_legend.text(100, 100+radius+10, "270°", ha='center', va='center', fontweight='bold', color='black')
    ax_legend.text(100-radius-10, 100, "180°", ha='center', va='center', fontweight='bold', color='black')
    
    # Add diagonal line for magnitude and orientation with arrows
    center = 100  # Center of the wheel
    arrow_length = 60  # Length of each arrow from center
    
    # Draw arrows from center in both directions
    ax_legend.arrow(center, center, arrow_length*0.7, -arrow_length*0.7, 
                   head_width=8, head_length=10, fc='k', ec='k', 
                   linewidth=2, length_includes_head=True)
    
    # Add curved text for orientation along the line
    import matplotlib.patheffects as path_effects
    
    # Create a path for the text to follow
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    
    # Add "Magnitude" text along the line
    ax_legend.text(100, 80, "Magnitude", ha='center', va='center', fontweight='bold', rotation=45, color='black')
    
    # Add "Orientation" text curved around the edge
    angle = 45  # Position at 45 degrees (top-right)
    x = 100 + 80 * np.cos(np.radians(angle))
    y = 100 - 80 * np.sin(np.radians(angle))
    ax_legend.text(x+20, y-20, "Orientation", ha='center', va='center', fontweight='bold',
                  rotation=-45, color='black')
    
    ax_legend.axis('off')
    
    # Create the blur field visualization
    # Normalize orientation to [0, 1] for Hue
    hue = (orientation_map + np.pi) / (2 * np.pi)
    
    # Normalize magnitude to [0, 1] for Saturation
    mag_max = np.max(magnitude_map) if np.max(magnitude_map) > 0 else 1.0
    saturation = magnitude_map / mag_max
    saturation = np.clip(saturation, 0, 1)
    
    # Value channel (brightness) at maximum
    value = np.ones_like(magnitude_map)
    
    # Stack H, S, V channels and convert to RGB
    hsv_image = np.stack([hue, saturation, value], axis=-1)
    
    # Convert HSV to RGB
    import matplotlib.colors as mcolors
    rgb_image = mcolors.hsv_to_rgb(hsv_image)
    
    # Load original image if provided
    original_image = None
    if image_path and os.path.exists(image_path):
        try:
            import cv2
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = cv2.resize(original_image, (W, H))
            
            # Blend original image with blur field
            alpha = 0.7  # Transparency of the blur field
            rgb_image = alpha * rgb_image + (1 - alpha) * original_image / 255.0
            rgb_image = np.clip(rgb_image, 0, 1)
            
        except Exception as e:
            print(f"Error loading original image: {e}")
    
    # Display the blur field visualization
    ax_vis = fig.add_subplot(gs[0, 1:])
    ax_vis.imshow(rgb_image)
    
    # Use the image name as the title (without .pt extension)
    image_title = os.path.basename(tensor_path)
    if image_title.endswith('.pt'):
        image_title = image_title[:-3]  # Remove .pt extension
    ax_vis.set_title(image_title, fontsize=12)
    ax_vis.axis('off')
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    return rgb_image

def visualize_blur_components(blur_data_path, image_path=None, output_path=None):
    """
    Visualize blur components (magnitude, x direction, y direction) from ground truth data.
    
    Args:
        blur_data_path: Path to the .npy file containing blur data (3, H, W)
        image_path: Optional path to the original image
        output_path: Path to save the visualization
    """
    # Load blur data
    if blur_data_path.endswith('.npy'):
        blur_data = np.load(blur_data_path)
    elif blur_data_path.endswith('.pt'):
        blur_data = torch.load(blur_data_path, map_location=torch.device('cpu'))
        if isinstance(blur_data, torch.Tensor):
            blur_data = blur_data.detach().cpu().numpy()
        else:
            raise TypeError(f"Expected tensor in .pt file, got {type(blur_data)}")
    else:
        raise ValueError(f"Unsupported file format: {blur_data_path}")
    
    # Ensure blur data has the right shape
    if blur_data.shape[0] != 3:
        raise ValueError(f"Expected blur data with 3 channels, got {blur_data.shape}")
    
    # Extract components
    x_direction = blur_data[0]
    y_direction = blur_data[1]
    magnitude = blur_data[2]
    
    # Load original image if provided
    original_image = None
    if image_path and os.path.exists(image_path):
        try:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            # Resize to match blur data dimensions
            original_image = cv2.resize(original_image, (x_direction.shape[1], x_direction.shape[0]))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            original_image = None
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Magnitude
    im_mag = axes[0].imshow(magnitude, cmap='viridis')
    axes[0].set_title('Blur Magnitude')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im_mag, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot 2: X Direction
    im_x = axes[1].imshow(x_direction, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('X Direction Blur')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im_x, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot 3: Y Direction
    im_y = axes[2].imshow(y_direction, cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set_title('Y Direction Blur')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im_y, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add overall title
    image_name = os.path.basename(blur_data_path)
    plt.suptitle(f'Blur Components: {image_name}', fontsize=16)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

def visualize_multiple_blur_fields(tensor_list, image_path_list=None, output_path=None):
    """
    Visualize multiple blur fields with a shared color wheel legend.
    
    Args:
        tensor_list: List of tensors or numpy arrays with shape (3, H, W) for bx, by, magnitude
        image_path_list: Optional list of paths to original images to blend with blur fields
        output_path: Path to save the visualization. If None, the figure will be displayed.
    
    Returns:
        Path to saved visualization if output_path is provided, otherwise None
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import torch
    import matplotlib.colors as mcolors
    
    # Create a figure with 6 subplots (1 for legend, 5 for images)
    fig = plt.figure(figsize=(20, 10))  # Increased height for more space
    
    # Create a grid layout with more space at the top
    gs = plt.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1], 
                     top=0.85)  # Reduced top to leave space for text
    
    # Create the color wheel for the legend
    wheel = create_color_wheel(size=300)
    
    # Add the color wheel legend in the first position
    ax_legend = fig.add_subplot(gs[0, 0])
    ax_legend.imshow(wheel)
    
    # Add title and explanation text ABOVE the color wheel using figure coordinates
    fig.text(0.25, 0.90, "Blur Field Color Legend", ha='center', va='center', 
             fontweight='bold', fontsize=16)
    
    # Add orientation labels to the color wheel
    center = 150  # Center of the wheel (half of size)
    radius = 160  # Slightly larger than wheel radius for labels
    ax_legend.text(center, center-radius-10, "90°", ha='center', va='center', fontweight='bold', color='black', fontsize=12)
    ax_legend.text(center+radius+10, center, "0°", ha='center', va='center', fontweight='bold', color='black', fontsize=12)
    ax_legend.text(center, center+radius+10, "270°", ha='center', va='center', fontweight='bold', color='black', fontsize=12)
    ax_legend.text(center-radius-10, center, "180°", ha='center', va='center', fontweight='bold', color='black', fontsize=12)
    
    # Add diagonal arrow for magnitude and orientation
    # Arrow pointing from center to top-right (45 degrees)
    arrow_length = 100
    dx = arrow_length * np.cos(np.radians(45))
    dy = -arrow_length * np.sin(np.radians(45))  # Negative because y-axis is inverted in images
    
    # Draw the arrow
    ax_legend.arrow(center, center, dx, dy, 
                   head_width=10, head_length=15, fc='black', ec='black', 
                   linewidth=2, length_includes_head=True)
    
    # Add "Magnitude" text along the arrow
    # Position text at 45 degrees, slightly offset from the arrow
    mag_x = center + 0.5 * dx
    mag_y = center + 0.5 * dy
    ax_legend.text(mag_x - 15, mag_y - 15, "Magnitude", ha='center', va='center', 
                  fontweight='bold', color='black', fontsize=12, rotation=45)
    
    # Add "Orientation" text curved around the edge
    # Position text near the edge at 45 degrees
    orient_x = center + 0.8 * dx
    orient_y = center + 0.8 * dy
    ax_legend.text(orient_x + 50, orient_y - 50, "Orientation", ha='center', va='center', 
                  fontweight='bold', color='black', fontsize=12, rotation=-45)
    
    ax_legend.axis('off')
    
    # Define positions for the 5 blur field visualizations
    positions = [
        gs[0, 1], gs[0, 2],  # Top row
        gs[1, 0], gs[1, 1], gs[1, 2]  # Bottom row
    ]
    
    # Process each tensor and create visualizations
    for i, tensor in enumerate(tensor_list[:5]):  # Limit to 5 images
        # Get the corresponding image path if available
        image_path = None
        if image_path_list and i < len(image_path_list):
            image_path = image_path_list[i]
        
        # Create subplot for this tensor
        ax = fig.add_subplot(positions[i])
        
        # Extract components
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = tensor
            
        bx = tensor_np[0]
        by = tensor_np[1]
        magnitude = tensor_np[2]
        
        # Calculate orientation
        orientation = np.arctan2(by, bx)
        
        # Create HSV representation
        hue = (orientation + np.pi) / (2 * np.pi)
        saturation = np.clip(magnitude / magnitude.max() if magnitude.max() > 0 else magnitude, 0, 1)
        value = np.ones_like(magnitude)
        
        # Stack HSV channels
        hsv = np.stack([hue, saturation, value], axis=-1)
        
        # Convert HSV to RGB
        rgb_image = mcolors.hsv_to_rgb(hsv)
        
        # If we have an original image, blend it with the blur field
        if image_path and os.path.exists(image_path):
            try:
                original_image = cv2.imread(image_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                original_image = cv2.resize(original_image, (bx.shape[1], bx.shape[0]))
                
                # Blend original image with blur field
                alpha = 0.7  # Transparency of the blur field
                rgb_image = alpha * rgb_image + (1 - alpha) * original_image / 255.0
                rgb_image = np.clip(rgb_image, 0, 1)
            except Exception as e:
                print(f"Error loading/blending image {image_path}: {e}")
        
        # Display the blur field
        ax.imshow(rgb_image)
        
        # Set title based on image path or index
        if image_path:
            title = os.path.basename(image_path)
            if len(title) > 20:  # Truncate long filenames
                title = title[:17] + "..."
        else:
            title = f"Blur Field {i+1}"
            
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Leave space at the top for titles
    
    # Save or display the figure
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_path}")
        return output_path
    else:
        plt.show()
        return None

def visualize_dataset_samples(dataset_path, output_dir="./visualizations", num_samples=5):
    """
    Visualize blur field samples from a dataset directory structure.
    
    Args:
        dataset_path: Path to dataset directory containing 'blur' and 'condition' subdirectories
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    
    Returns:
        Path to the multiple visualization file
    """
    import os
    import glob
    import numpy as np
    import torch
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paths to blur images and condition files
    blur_dir = os.path.join(dataset_path, 'blur')
    condition_dir = os.path.join(dataset_path, 'condition')
    
    if not os.path.exists(blur_dir):
        print(f"Error: Blur directory not found at {blur_dir}")
        return None
    
    if not os.path.exists(condition_dir):
        print(f"Error: Condition directory not found at {condition_dir}")
        return None
    
    # Get all blur images
    blur_images = sorted(glob.glob(os.path.join(blur_dir, '*.png')))
    if not blur_images:
        blur_images = sorted(glob.glob(os.path.join(blur_dir, '*.jpg')))
    
    if not blur_images:
        print(f"No image files found in {blur_dir}")
        return None
    
    # Get all condition files
    condition_files = sorted(glob.glob(os.path.join(condition_dir, '*.npy')))
    
    if not condition_files:
        print(f"No .npy files found in {condition_dir}")
        return None
    
    print(f"Found {len(blur_images)} blur images and {len(condition_files)} condition files")
    
    # Match blur images with condition files
    image_path_list = []
    tensor_list = []
    
    # Take the first num_samples
    for i in range(min(num_samples, len(blur_images), len(condition_files))):
        blur_image = blur_images[i]
        condition_file = condition_files[i]
        
        # Load the condition tensor
        try:
            tensor = np.load(condition_file)
            tensor_list.append(tensor)
            image_path_list.append(blur_image)
            
            # Save a copy of the tensor as .pt for compatibility
            tensor_path = os.path.join(output_dir, f"sample_{i}_flow.pt")
            torch.save(torch.from_numpy(tensor), tensor_path)
            
            print(f"Loaded sample {i+1}: {os.path.basename(blur_image)} and {os.path.basename(condition_file)}")
            
            # Create individual visualization for this sample
            try:
                # Generate output path for this individual visualization
                components_vis_path = os.path.join(output_dir, f"sample_{i}_components.png")
                
                # Call the visualization function that shows magnitude, x direction, y direction
                visualize_blur_components(
                    condition_file,  # Use the .npy file directly
                    image_path=blur_image,
                    output_path=components_vis_path
                )
                
                print(f"Created components visualization for sample {i+1} at {components_vis_path}")
                
                # Also create the color wheel visualization
                color_vis_output_path = os.path.join(output_dir, f"sample_{i}_color_visualization.png")
                
                visualize_blur_field_with_legend(
                    tensor_path=tensor_path,
                    image_path=blur_image,
                    output_path=color_vis_output_path,
                    title=f"Blur Field - {os.path.basename(blur_image)}"
                )
                
                print(f"Created color wheel visualization for sample {i+1} at {color_vis_output_path}")
                
            except Exception as e:
                print(f"Error creating visualizations for sample {i+1}: {e}")
                
        except Exception as e:
            print(f"Error loading condition file {condition_file}: {e}")
    
    if not tensor_list:
        print("No valid samples found")
        return None
    
    # Create multiple visualization
    try:
        output_path = os.path.join(output_dir, "multiple_blur_fields.png")
        visualize_multiple_blur_fields(tensor_list, image_path_list, output_path)
        print("Multiple visualization created successfully")
        return output_path
    except Exception as e:
        print(f"Error creating multiple visualization: {e}")
        return None

# Add a command-line interface to make the file directly executable
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize blur fields from dataset or individual files")
    parser.add_argument("--dataset", type=str, help="Path to dataset directory with 'blur' and 'condition' subdirectories")
    parser.add_argument("--tensor_paths", nargs='+', help="Paths to individual tensor files (.pt or .npy)")
    parser.add_argument("--image_paths", nargs='+', help="Paths to corresponding images (optional)")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize from dataset")
    parser.add_argument("--single_tensor", type=str, help="Path to a single tensor file to visualize with detailed components")
    parser.add_argument("--single_image", type=str, help="Path to corresponding image for single tensor visualization")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset:
        # Visualize samples from dataset
        visualize_dataset_samples(args.dataset, args.output_dir, args.num_samples)
    
    elif args.tensor_paths:
        # Visualize multiple individual tensors
        tensor_list = []
        for tensor_path in args.tensor_paths:
            if tensor_path.endswith('.npy'):
                tensor = np.load(tensor_path)
            elif tensor_path.endswith('.pt'):
                tensor = torch.load(tensor_path, map_location=torch.device('cpu'))
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.detach().cpu().numpy()
            else:
                print(f"Unsupported file format: {tensor_path}")
                continue
            tensor_list.append(tensor)
        
        # Get corresponding image paths if provided
        image_path_list = args.image_paths if args.image_paths else None
        
        # Create visualization
        output_path = os.path.join(args.output_dir, "multiple_blur_fields.png")
        visualize_multiple_blur_fields(tensor_list, image_path_list, output_path)
    
    elif args.single_tensor:
        # Visualize a single tensor with detailed components
        if args.single_tensor.endswith('.npy') or args.single_tensor.endswith('.pt'):
            output_path = os.path.join(args.output_dir, "blur_components.png")
            visualize_blur_components(args.single_tensor, args.single_image, output_path)
            
            # Also create color wheel visualization
            color_output_path = os.path.join(args.output_dir, "blur_field_color.png")
            visualize_blur_field_with_legend(args.single_tensor, args.single_image, color_output_path)
        else:
            print(f"Unsupported file format: {args.single_tensor}")
    
    else:
        parser.print_help()
