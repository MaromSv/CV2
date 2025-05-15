import torch
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

def create_direction_wheel(ax, title="Direction Color Wheel"):
    """Create a color wheel to show the mapping between direction and color"""
    # Create a circle with colored segments
    n = 100  # Number of segments
    theta = np.linspace(0, 2*np.pi, n)
    r = np.ones_like(theta)
    
    # Create a circular grid
    xx = r * np.cos(theta)
    yy = r * np.sin(theta)
    
    # Convert to HSV color space (hue represents direction)
    hue = (np.arctan2(yy, xx) / (2*np.pi)) % 1.0
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    
    # Convert HSV to RGB
    hsv = np.dstack((hue, sat, val))
    rgb = mcolors.hsv_to_rgb(hsv)
    
    # Plot the wheel
    ax.scatter(xx, yy, c=rgb.reshape(-1, 3), s=50)
    
    # Add a circle outline
    circle = Circle((0, 0), 1, fill=False, edgecolor='black')
    ax.add_patch(circle)
    
    # Add direction labels
    ax.text(0, 1.1, "Up", ha='center', va='center')
    ax.text(0, -1.1, "Down", ha='center', va='center')
    ax.text(1.1, 0, "Right", ha='center', va='center')
    ax.text(-1.1, 0, "Left", ha='center', va='center')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')

def visualize_blur_field_label(label_path, image_path, output_dir):
    """Visualize blur field labels from .npy file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the label file
    try:
        blur_field = np.load(label_path)
        print(f"Loaded blur field with shape: {blur_field.shape}")
        
        # Check if the blur field has the expected format
        if blur_field.ndim != 3 or blur_field.shape[0] not in [2, 3]:
            print(f"Warning: Expected blur field with shape (2 or 3, H, W), got {blur_field.shape}")
    except Exception as e:
        print(f"Error loading blur field from {label_path}: {e}")
        return
    
    # Load the corresponding image if provided
    original_img = None
    if image_path and os.path.exists(image_path):
        try:
            original_img = Image.open(image_path).convert('RGB')
            print(f"Loaded image with size: {original_img.size}")
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
    
    # Extract dx, dy, and magnitude if available
    if blur_field.shape[0] == 2:
        # Format is [dx, dy]
        dx = blur_field[0]
        dy = blur_field[1]
        # Calculate magnitude
        mag = np.sqrt(dx**2 + dy**2)
        print("Blur field format: [dx, dy] - calculated magnitude")
    elif blur_field.shape[0] == 3:
        # Format is [dx, dy, magnitude]
        dx = blur_field[0]
        dy = blur_field[1]
        mag = blur_field[2]
        print("Blur field format: [dx, dy, magnitude]")
    
    # Create base filename for outputs
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot original image if available
    if original_img:
        axs[0, 0].imshow(np.array(original_img))
        axs[0, 0].set_title('Original Image')
    else:
        axs[0, 0].set_title('Original Image (Not Available)')
    axs[0, 0].axis('off')
    
    # Plot magnitude as heatmap
    mag_plot = axs[0, 1].imshow(mag, cmap='viridis')
    axs[0, 1].set_title('Blur Magnitude')
    axs[0, 1].axis('off')
    plt.colorbar(mag_plot, ax=axs[0, 1])
    
    # Plot dx
    dx_plot = axs[1, 0].imshow(dx, cmap='coolwarm')
    axs[1, 0].set_title('Displacement X')
    axs[1, 0].axis('off')
    plt.colorbar(dx_plot, ax=axs[1, 0])
    
    # Plot dy
    dy_plot = axs[1, 1].imshow(dy, cmap='coolwarm')
    axs[1, 1].set_title('Displacement Y')
    axs[1, 1].axis('off')
    plt.colorbar(dy_plot, ax=axs[1, 1])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{base_name}_components.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Create a second visualization with vectors
    plt.figure(figsize=(10, 8))
    plt.imshow(mag, cmap='viridis')
    plt.colorbar(label='Blur Magnitude')
    
    # Downsample for clearer visualization
    h, w = mag.shape
    step = max(1, min(h, w) // 30)  # Adjust step size based on image dimensions
    
    # Create grid for quiver plot
    y, x = np.mgrid[0:h:step, 0:w:step]
    dx_down = dx[::step, ::step]
    dy_down = dy[::step, ::step]
    
    # Plot direction vectors
    plt.quiver(x, y, dx_down, dy_down, color='white', scale=1.0, width=0.002)
    
    plt.title('Blur Field Direction Visualization')
    plt.axis('off')
    plt.tight_layout()
    vector_path = os.path.join(output_dir, f"{base_name}_vectors.png")
    plt.savefig(vector_path, dpi=300, bbox_inches='tight')
    
    # Create direction-color visualization
    # Calculate direction angle in radians
    angle = np.arctan2(dy, dx)
    
    # Normalize angle to [0, 1] for hue
    hue = (angle / (2 * np.pi)) % 1.0
    
    # Normalize magnitude to [0, 1] for value/brightness
    mag_max = np.max(mag) if np.max(mag) > 0 else 1.0
    mag_normalized = mag / mag_max
    
    # Create HSV image (hue=direction, saturation=1, value=magnitude)
    hsv = np.zeros((hue.shape[0], hue.shape[1], 3))
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 1.0  # Full saturation
    hsv[:, :, 2] = mag_normalized
    
    # Convert HSV to RGB
    rgb = mcolors.hsv_to_rgb(hsv)
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    # Plot the direction-colored image
    ax1 = fig.add_subplot(121)
    ax1.imshow(rgb)
    ax1.set_title("Color = Direction, Brightness = Magnitude")
    ax1.axis('off')
    
    # Add the direction color wheel as legend
    ax2 = fig.add_subplot(122)
    create_direction_wheel(ax2)
    
    plt.tight_layout()
    color_path = os.path.join(output_dir, f"{base_name}_direction_color.png")
    plt.savefig(color_path, dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {output_dir}")
    print(f"- Components: {save_path}")
    print(f"- Vectors: {vector_path}")
    print(f"- Direction color: {color_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", required=True, type=str, help="Path to blur field label (.npy file)")
    parser.add_argument("--image_path", default=None, type=str, help="Path to corresponding image (optional)")
    parser.add_argument("--output_dir", default="./label_results", type=str, help="Output directory")
    args = parser.parse_args()
    
    visualize_blur_field_label(args.label_path, args.image_path, args.output_dir)