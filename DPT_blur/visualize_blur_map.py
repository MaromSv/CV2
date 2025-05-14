import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2 # Import OpenCV

def visualize_blur_map(tensor_path, image_path=None, quiver_step=16):
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
        # Load the tensor
        blur_map_tensor = torch.load(tensor_path, map_location=torch.device('cpu')) # Ensure load to CPU
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
    magnitude_map = blur_map_np[2, :, :] # Magnitude is now direct from channel 2
    H, W = bx.shape

    # Calculate Orientation (Angle) from bx and by
    orientation_map = np.arctan2(by, bx) # Angle in radians [-pi, pi]

    print(f"Extracted bx - Min: {bx.min():.4f}, Max: {bx.max():.4f}, Mean: {bx.mean():.4f}")
    print(f"Extracted by - Min: {by.min():.4f}, Max: {by.max():.4f}, Mean: {by.mean():.4f}")
    print(f"Extracted Magnitude map (Channel 2) - Min: {magnitude_map.min():.4f}, Max: {magnitude_map.max():.4f}, Mean: {magnitude_map.mean():.4f}")
    print(f"Calculated Orientation map (radians) - Min: {orientation_map.min():.4f}, Max: {orientation_map.max():.4f}, Mean: {orientation_map.mean():.4f}")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Increased height slightly for titles
    plt.subplots_adjust(top=0.85) # Adjust top to make space for suptitle

    # --- Plot 1: Colorwheel representation (Orientation as Hue, Magnitude as Saturation) ---
    # Normalize orientation to [0, 1] for Hue
    hue_channel = (orientation_map + np.pi) / (2 * np.pi)

    # Normalize magnitude to [0, 1] for Saturation
    min_mag, max_mag = np.min(magnitude_map), np.max(magnitude_map)
    if max_mag == min_mag:
        saturation_channel = np.zeros_like(magnitude_map) # Avoid division by zero, set to 0 saturation
    else:
        saturation_channel = (magnitude_map - min_mag) / (max_mag - min_mag)
    saturation_channel = np.clip(saturation_channel, 0, 1)

    # Value channel (brightness) at maximum
    value_channel = np.ones_like(magnitude_map)

    # Stack H, S, V channels and convert to RGB
    # HSV image shape: (H, W, 3)
    hsv_colorwheel_image = np.stack([hue_channel, saturation_channel, value_channel], axis=-1)
    
    # Ensure matplotlib.colors is available, usually plt.matplotlib.colors
    try:
        import matplotlib.colors
        rgb_colorwheel_image = matplotlib.colors.hsv_to_rgb(hsv_colorwheel_image)
    except ImportError:
        # Fallback or error if direct import fails - though plt should have it
        print("Warning: matplotlib.colors could not be imported directly. Trying plt.cm.hsv as a limited alternative if needed.")
        # This fallback is not ideal as plt.cm.hsv expects a single channel for hue and S,V are fixed.
        # The primary approach using matplotlib.colors.hsv_to_rgb is preferred.
        # For this specific case, we must construct the HSV image first, so this fallback isn't used here.
        # The try-except is more for illustrating where it might be needed if not using plt.
        # We will assume matplotlib.colors.hsv_to_rgb works.
        rgb_colorwheel_image = plt.matplotlib.colors.hsv_to_rgb(hsv_colorwheel_image)


    axes[0].imshow(rgb_colorwheel_image)
    axes[0].set_title('Blur Field (Angle-Hue, Mag-Sat)')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # --- Plot 2: Magnitude (Moved from original axes[0]) ---
    im_mag = axes[1].imshow(magnitude_map, cmap='viridis')
    axes[1].set_title('Calculated Blur Magnitude')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im_mag, ax=axes[1], fraction=0.046, pad=0.04) # Associated with axes[1]

    # --- Plot 3: Vector Field (Quiver) ---
    # Create coordinates
    x_coords = np.arange(0, W)
    y_coords = np.arange(0, H)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Subsample for clarity
    step = max(1, quiver_step) # Ensure step is at least 1
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    bx_sub = bx[::step, ::step]
    by_sub = by[::step, ::step]
    magnitude_sub = magnitude_map[::step, ::step] # Subsample magnitude for coloring vectors

    # Display original image if available, otherwise a gray background
    if original_image is not None:
        # Ensure image dimensions match blur map if necessary. Assuming they do for now.
        if original_image.shape[0] != H or original_image.shape[1] != W:
            print(f"Warning: Original image dimensions ({original_image.shape[0]}x{original_image.shape[1]}) ")
            print(f"do not match blur map dimensions ({H}x{W}). Resizing image for visualization.")
            original_image_resized = cv2.resize(original_image, (W, H), interpolation=cv2.INTER_AREA)
            axes[2].imshow(original_image_resized)
        else:
            axes[2].imshow(original_image)
    else:
        dummy_image = np.ones((H, W)) * 0.5
        axes[2].imshow(dummy_image, cmap='gray', vmin=0, vmax=1)

    # Plot vectors using quiver, colored by magnitude
    # scale=None lets matplotlib autoscale, adjust 'width' and 'color' as needed
    quiv = axes[2].quiver(X_sub, Y_sub, bx_sub, -by_sub, # Invert by for image coords (y down)
                          magnitude_sub, # Color by subsampled magnitude
                          cmap='viridis', # Use a colormap for magnitudes
                          scale=None, scale_units='xy', angles='xy', width=0.003,
                          pivot='middle') # Pivot arrows from their center
    axes[2].set_title(f'Predicted Blur Vector Field (Step={step})')
    axes[2].set_aspect('equal', adjustable='box') # Ensure aspect ratio is correct
    axes[2].set_xlim(0, W)
    axes[2].set_ylim(H, 0) # Invert y-axis for image display
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    # Add a colorbar for the quiver plot
    fig.colorbar(quiv, ax=axes[2], fraction=0.046, pad=0.04, label='Vector Magnitude')

    plt.suptitle(f'Blur Vector Visualization ({os.path.basename(tensor_path)})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjusted rect to accommodate suptitle with subplots_adjust
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a saved blur vector tensor (bx, by) by calculating and showing magnitude, orientation, and the vector field itself, optionally superimposed on the original image.")
    parser.add_argument('--input_file', type=str, default='results/output_blur_map.pt', help='Path to the saved blur vector tensor (.pt file).')
    parser.add_argument('--image_path', type=str, default='data/GOPRO_Large/test/GOPR0384_11_00/blur/000001.png', help='Optional path to the original input image to superimpose the vector field on.')
    parser.add_argument('--step', type=int, default=16, help='Step size for quiver plot subsampling.')
    args = parser.parse_args()

    visualize_blur_map(args.input_file, image_path=args.image_path, quiver_step=args.step) 