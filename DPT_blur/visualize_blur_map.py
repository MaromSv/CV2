import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def visualize_blur_map(tensor_path, quiver_step=16):
    """Loads a saved blur vector (bx, by) tensor and visualizes its calculated magnitude, orientation, and the vector field itself."""

    if not os.path.exists(tensor_path):
        print(f"Error: File not found at {tensor_path}")
        return

    try:
        # Load the tensor
        blur_vector_map = torch.load(tensor_path, map_location=torch.device('cpu')) # Ensure load to CPU
        print(f"Loaded tensor from {tensor_path} with shape: {blur_vector_map.shape}")

        if not isinstance(blur_vector_map, torch.Tensor):
             raise TypeError("Loaded object is not a torch.Tensor")
        if blur_vector_map.dim() != 3 or blur_vector_map.shape[0] != 2:
            raise ValueError(f"Expected tensor shape (2, H, W) for (bx, by), but got {blur_vector_map.shape}")

        # Detach and convert to numpy
        blur_vector_np = blur_vector_map.detach().cpu().numpy()

    except Exception as e:
        print(f"Error loading or processing tensor from {tensor_path}: {e}")
        return

    # Extract components
    bx = blur_vector_np[0, :, :]
    by = blur_vector_np[1, :, :]
    H, W = bx.shape

    # Calculate Magnitude and Orientation (Angle)
    magnitude_map = np.sqrt(bx**2 + by**2)
    orientation_map = np.arctan2(by, bx) # Angle in radians [-pi, pi]

    print(f"Calculated Magnitude map - Min: {magnitude_map.min():.4f}, Max: {magnitude_map.max():.4f}, Mean: {magnitude_map.mean():.4f}")
    print(f"Calculated Orientation map (radians) - Min: {orientation_map.min():.4f}, Max: {orientation_map.max():.4f}, Mean: {orientation_map.mean():.4f}")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Changed to 3 subplots

    # --- Plot 1: Magnitude ---
    im_mag = axes[0].imshow(magnitude_map, cmap='viridis')
    axes[0].set_title('Calculated Blur Magnitude')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im_mag, ax=axes[0], fraction=0.046, pad=0.04)

    # --- Plot 2: Orientation ---
    orientation_normalized = (orientation_map + np.pi) / (2 * np.pi)
    orientation_normalized = np.clip(orientation_normalized, 0, 1)
    rgb_image = plt.cm.hsv(orientation_normalized)
    im_ori = axes[1].imshow(rgb_image[..., :3])
    axes[1].set_title('Calculated Blur Orientation (Angle -> Color)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

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

    # Create a dummy background (gray)
    dummy_image = np.ones((H, W)) * 0.5
    axes[2].imshow(dummy_image, cmap='gray', vmin=0, vmax=1)

    # Plot vectors using quiver
    # scale=None lets matplotlib autoscale, adjust 'width' and 'color' as needed
    axes[2].quiver(X_sub, Y_sub, bx_sub, -by_sub, # Invert by for image coords (y down)
                   color='red', scale=None, scale_units='xy', angles='xy', width=0.003)
    axes[2].set_title(f'Predicted Blur Vector Field (Step={step})')
    axes[2].set_aspect('equal', adjustable='box') # Ensure aspect ratio is correct
    axes[2].set_xlim(0, W)
    axes[2].set_ylim(H, 0) # Invert y-axis for image display
    axes[2].set_xticks([])
    axes[2].set_yticks([])


    plt.suptitle(f'Blur Vector Visualization ({os.path.basename(tensor_path)})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a saved blur vector tensor (bx, by) by calculating and showing magnitude, orientation, and the vector field.")
    parser.add_argument('input_file', type=str, help='Path to the saved blur vector tensor (.pt file).')
    parser.add_argument('--step', type=int, default=16, help='Step size for quiver plot subsampling.')
    args = parser.parse_args()

    visualize_blur_map(args.input_file, quiver_step=args.step) 