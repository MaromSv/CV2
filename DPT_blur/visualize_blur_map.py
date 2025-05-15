import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2 # Import OpenCV

def derive_gt_path(original_image_path_from_pt_file):
    """Derives the GT .npy path from the original image path.
    Assumes structure like: .../<split>/blur/image.png -> .../<split>/condition/image.npy
    """
    if not original_image_path_from_pt_file:
        return None
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

    plt.suptitle(f'Blur Comparison ({os.path.basename(predicted_tensor_path)})', fontsize=16)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predicted and GT blur maps.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the saved PREDICTED blur data tensor (.pt file from inference).')
    parser.add_argument('--image_path', type=str, default=None, help='Optional path to the original input image. Overrides path in .pt file.')
    parser.add_argument('--step', type=int, default=16, help='Step size for quiver plot subsampling.')
    args = parser.parse_args()

    visualize_blur_map(args.input_file, image_path_cli=args.image_path, quiver_step=args.step) 