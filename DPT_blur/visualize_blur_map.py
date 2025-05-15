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
    
    orientation_for_hsv = np.arctan2(sin_comp, cos_comp) # For HSV plot
    bx_for_quiver = mag_map * cos_comp # For quiver plot
    by_for_quiver = mag_map * sin_comp # For quiver plot

    print(f"\n--- {data_label} --- ")
    print(f"  Cos Comp (Ch0) - Min: {cos_comp.min():.4f}, Max: {cos_comp.max():.4f}, Mean: {cos_comp.mean():.4f}")
    print(f"  Sin Comp (Ch1) - Min: {sin_comp.min():.4f}, Max: {sin_comp.max():.4f}, Mean: {sin_comp.mean():.4f}")
    print(f"  Magnitude (Ch2) - Min: {mag_map.min():.4f}, Max: {mag_map.max():.4f}, Mean: {mag_map.mean():.4f}")
    print(f"  bx Quiver (Mag*Cos) - Min: {bx_for_quiver.min():.4f}, Max: {bx_for_quiver.max():.4f}, Mean: {bx_for_quiver.mean():.4f}")
    print(f"  by Quiver (Mag*Sin) - Min: {by_for_quiver.min():.4f}, Max: {by_for_quiver.max():.4f}, Mean: {by_for_quiver.mean():.4f}")
    return cos_comp, sin_comp, mag_map, orientation_for_hsv, bx_for_quiver, by_for_quiver

def plot_visualization_row(axes_row, cos_comp, sin_comp, mag_map, orientation_hsv, bx_quiver, by_quiver, 
                           original_image_display, quiver_step, H, W, row_title_prefix=""):
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
    x_coords, y_coords = np.meshgrid(np.arange(0, W), np.arange(0, H))
    step = max(1, quiver_step)
    X_sub, Y_sub = x_coords[::step, ::step], y_coords[::step, ::step]
    bx_sub, by_sub = bx_quiver[::step, ::step], by_quiver[::step, ::step]
    mag_sub_for_color = mag_map[::step, ::step]

    if original_image_display is not None:
        if original_image_display.shape[0] != H or original_image_display.shape[1] != W:
            original_image_resized = cv2.resize(original_image_display, (W, H), interpolation=cv2.INTER_AREA)
            axes_row[2].imshow(original_image_resized)
        else:
            axes_row[2].imshow(original_image_display)
    else:
        axes_row[2].imshow(np.ones((H, W)) * 0.5, cmap='gray', vmin=0, vmax=1)
    
    quiv = axes_row[2].quiver(X_sub, Y_sub, bx_sub, -by_sub, mag_sub_for_color,
                              cmap='viridis', scale=None, scale_units='xy', 
                              angles='xy', width=0.003, pivot='middle')
    axes_row[2].set_title(f'{row_title_prefix} Vector Field (Step={step})')
    axes_row[2].set_aspect('equal', adjustable='box'); axes_row[2].set_xlim(0, W); axes_row[2].set_ylim(H, 0)
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
        loaded_data = torch.load(predicted_tensor_path, map_location=torch.device('cpu'))
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

    cos_pred, sin_pred, mag_pred, orientation_hsv_pred, bx_quiver_pred, by_quiver_pred = \
        process_blur_data(pred_blur_map_np, "Prediction")
    H_pred, W_pred = mag_pred.shape

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
        cos_gt, sin_gt, mag_gt, orientation_hsv_gt, bx_quiver_gt, by_quiver_gt = \
            process_blur_data(gt_map_np, "Ground Truth")
        plot_visualization_row(axes[0,:], cos_gt, sin_gt, mag_gt, orientation_hsv_gt, 
                               bx_quiver_gt, by_quiver_gt, original_image_for_display, 
                               quiver_step, H_pred, W_pred, "GT") # Assume GT matches pred dimensions for now
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
