import torch
import torch.nn as nn
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from MIMOUNet import build_MIMOUnet_net

# Define global variable for the original image
original_img = None

def create_direction_wheel_with_magnitude(ax, title="Blur Condition Field"):
    """Create a color wheel to show the mapping between direction and color with magnitude indicator"""
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
    circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    
    # Add direction labels with degrees
    ax.text(0, 1.2, "90°", ha='center', va='center', fontweight='bold')
    ax.text(0, -1.2, "270°", ha='center', va='center', fontweight='bold')
    ax.text(1.2, 0, "0°", ha='center', va='center', fontweight='bold')
    ax.text(-1.2, 0, "180°", ha='center', va='center', fontweight='bold')
    
    # Add magnitude arrow
    ax.arrow(0, 0, 0.7, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax.text(0.35, 0.15, "Magnitude", ha='center', va='center', fontweight='bold')
    ax.text(0.35, -0.15, "→", ha='center', va='center', fontweight='bold')
    
    # Add orientation label
    ax.text(0, 0.5, "Orientation", ha='center', va='center', rotation=90, fontweight='bold')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.axis('off')

def visualize_blur_field(dx, dy, mag, save_path, title_prefix=""):
    """Create a visualization of the blur field using arrows"""
    # Convert tensors to numpy arrays
    dx_np = dx.squeeze().cpu().numpy()
    dy_np = dy.squeeze().cpu().numpy()
    mag_np = mag.squeeze().cpu().numpy()
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot original image
    axs[0, 0].imshow(np.array(original_img))
    axs[0, 0].set_title(f'{title_prefix}Original Blurred Image')
    axs[0, 0].axis('off')
    
    # Plot magnitude as heatmap
    mag_plot = axs[0, 1].imshow(mag_np, cmap='viridis')
    axs[0, 1].set_title(f'{title_prefix}Blur Magnitude')
    axs[0, 1].axis('off')
    plt.colorbar(mag_plot, ax=axs[0, 1])
    
    # Plot dx
    dx_plot = axs[1, 0].imshow(dx_np, cmap='coolwarm')
    axs[1, 0].set_title(f'{title_prefix}Displacement X')
    axs[1, 0].axis('off')
    plt.colorbar(dx_plot, ax=axs[1, 0])
    
    # Plot dy
    dy_plot = axs[1, 1].imshow(dy_np, cmap='coolwarm')
    axs[1, 1].set_title(f'{title_prefix}Displacement Y')
    axs[1, 1].axis('off')
    plt.colorbar(dy_plot, ax=axs[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Create a second visualization with vectors
    plt.figure(figsize=(10, 8))
    plt.imshow(mag_np, cmap='viridis')
    plt.colorbar(label='Blur Magnitude')
    
    # Downsample for clearer visualization
    h, w = mag_np.shape
    step = max(1, min(h, w) // 30)  # Adjust step size based on image dimensions
    
    # Create grid for quiver plot
    y, x = np.mgrid[0:h:step, 0:w:step]
    dx_down = dx_np[::step, ::step]
    dy_down = dy_np[::step, ::step]
    
    # Plot direction vectors
    plt.quiver(x, y, dx_down, dy_down, color='white', scale=1.0, width=0.002)
    
    plt.title(f'{title_prefix}Blur Field Direction Visualization')
    plt.axis('off')
    plt.tight_layout()
    vector_path = save_path.replace('.png', '_vectors.png')
    plt.savefig(vector_path, dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {save_path} and {vector_path}")

def visualize_direction_color(dx, dy, mag, img, save_path, title):
    """Create a visualization where color represents direction and intensity represents magnitude"""
    # Convert tensors to numpy arrays
    dx_np = dx.squeeze().cpu().numpy()
    dy_np = dy.squeeze().cpu().numpy()
    mag_np = mag.squeeze().cpu().numpy()
    
    # Calculate direction angle in radians
    angle = np.arctan2(dy_np, dx_np)
    
    # Normalize angle to [0, 1] for hue
    hue = (angle / (2 * np.pi)) % 1.0
    
    # Normalize magnitude to [0, 1] for value/brightness
    # Clip magnitude to avoid extreme values
    mag_normalized = np.clip(mag_np, -0.5, 0.5)
    mag_normalized = (mag_normalized + 0.5)  # Now in range [0, 1]
    
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
    ax1.set_title(f"{title}\nColor = Direction, Brightness = Magnitude")
    ax1.axis('off')
    
    # Add the direction color wheel as legend
    ax2 = fig.add_subplot(122)
    create_direction_wheel(ax2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Direction-color visualization saved to {save_path}")

def visualize_all_resolutions(dx_list, dy_list, mag_list, img_list, res_names, save_path):
    """Create a combined visualization of all resolution levels"""
    # Create figure with 4 rows (original images + 3 resolution levels)
    fig, axs = plt.subplots(4, 3, figsize=(15, 16))
    
    # First row: original images at different resolutions
    for i, (img, name) in enumerate(zip(img_list, res_names)):
        axs[0, i].imshow(np.array(img))
        axs[0, i].set_title(f"{name.replace('_', ' ').title()}: Original Image")
        axs[0, i].axis('off')
    
    # For each resolution level
    for i, (dx, dy, mag, name) in enumerate(zip(dx_list, dy_list, mag_list, res_names)):
        # Convert tensors to numpy arrays
        dx_np = dx.squeeze().cpu().numpy()
        dy_np = dy.squeeze().cpu().numpy()
        mag_np = mag.squeeze().cpu().numpy()
        
        # Calculate direction angle in radians
        angle = np.arctan2(dy_np, dx_np)
        
        # Normalize angle to [0, 1] for hue
        hue = (angle / (2 * np.pi)) % 1.0
        
        # Normalize magnitude to [0, 1] for value/brightness
        mag_normalized = np.clip(mag_np, -0.5, 0.5)
        mag_normalized = (mag_normalized + 0.5)  # Now in range [0, 1]
        
        # Create HSV image (hue=direction, saturation=1, value=magnitude)
        hsv = np.zeros((hue.shape[0], hue.shape[1], 3))
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = 1.0  # Full saturation
        hsv[:, :, 2] = mag_normalized
        
        # Convert HSV to RGB
        rgb = mcolors.hsv_to_rgb(hsv)
        
        # Plot direction-colored image
        axs[1, i].imshow(rgb)
        axs[1, i].set_title(f"{name.replace('_', ' ').title()}: Direction Color")
        axs[1, i].axis('off')
        
        # Plot magnitude
        mag_plot = axs[2, i].imshow(mag_np, cmap='viridis')
        axs[2, i].set_title(f"{name.replace('_', ' ').title()}: Magnitude")
        axs[2, i].axis('off')
        if i == 2:  # Add colorbar only for the last column
            plt.colorbar(mag_plot, ax=axs[2, i])
        
        # Plot direction vectors
        axs[3, i].imshow(mag_np, cmap='viridis', alpha=0.7)
        
        # Downsample for clearer visualization
        h, w = mag_np.shape
        step = max(1, min(h, w) // 20)  # Adjust step size based on image dimensions
        
        # Create grid for quiver plot
        y, x = np.mgrid[0:h:step, 0:w:step]
        dx_down = dx_np[::step, ::step]
        dy_down = dy_np[::step, ::step]
        
        # Plot direction vectors
        axs[3, i].quiver(x, y, dx_down, dy_down, color='white', scale=1.0, width=0.002)
        axs[3, i].set_title(f"{name.replace('_', ' ').title()}: Direction Vectors")
        axs[3, i].axis('off')
    
    # Add a color wheel as a legend in a separate figure
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    create_direction_wheel(ax, "Direction Color Legend")
    plt.tight_layout()
    legend_path = save_path.replace('.png', '_legend.png')
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    
    # Save the main figure
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined visualization saved to {save_path} and {legend_path}")

def visualize_blur_conditions(dx_list, dy_list, mag_list, save_path, condition_names=None):
    """Create a visualization similar to the reference image with color wheel and conditions"""
    if condition_names is None:
        condition_names = [f"Condition {chr(65+i)}" for i in range(len(dx_list))]
    
    # Create figure with color wheel on left and conditions on right
    fig = plt.figure(figsize=(15, 5))
    
    # Create grid spec for layout
    gs = plt.GridSpec(1, len(dx_list) + 1, width_ratios=[1] + [1] * len(dx_list))
    
    # Add color wheel
    ax_wheel = fig.add_subplot(gs[0, 0])
    create_direction_wheel_with_magnitude(ax_wheel, title="Blur Condition Field")
    
    # Process each condition
    for i, (dx, dy, mag, name) in enumerate(zip(dx_list, dy_list, mag_list, condition_names)):
        # Convert tensors to numpy arrays
        dx_np = dx.squeeze().cpu().numpy()
        dy_np = dy.squeeze().cpu().numpy()
        mag_np = mag.squeeze().cpu().numpy()
        
        # Calculate direction angle in radians
        angle = np.arctan2(dy_np, dx_np)
        
        # Normalize angle to [0, 1] for hue
        hue = (angle / (2 * np.pi)) % 1.0
        
        # Normalize magnitude to [0, 1] for value/brightness
        mag_normalized = np.clip(mag_np, -0.5, 0.5)
        mag_normalized = (mag_normalized + 0.5)  # Now in range [0, 1]
        
        # Create HSV image (hue=direction, saturation=1, value=magnitude)
        hsv = np.zeros((hue.shape[0], hue.shape[1], 3))
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = 1.0  # Full saturation
        hsv[:, :, 2] = mag_normalized
        
        # Convert HSV to RGB
        rgb = mcolors.hsv_to_rgb(hsv)
        
        # Add condition subplot
        ax_cond = fig.add_subplot(gs[0, i+1])
        ax_cond.imshow(rgb)
        ax_cond.set_title(name, fontweight='bold', fontsize=14)
        ax_cond.axis('off')
    
    # Add arrows connecting wheel to conditions
    for i in range(len(dx_list)):
        fig.text(0.25 + i*0.2, 0.05, "↓", ha='center', va='center', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Blur conditions visualization saved to {save_path}")

@torch.no_grad()
def test_blur_field(image_path, output_dir, device='cuda'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model (without loading weights)
    model_name = 'MIMO-UNetPlus'  # or 'MIMO-UNet'
    net = build_MIMOUnet_net(model_name)
    net = nn.DataParallel(net)
    net.to(device)
    net.eval()
    
    # Load and preprocess image
    global original_img
    original_img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # Get original dimensions
    b, c, h, w = input_tensor.shape
    
    # Pad to multiple of 8 if needed
    factor = 8
    h_n = (factor - h % factor) % factor
    w_n = (factor - w % factor) % factor
    input_tensor = torch.nn.functional.pad(input_tensor, (0, w_n, 0, h_n), mode='reflect')
    
    # Forward pass
    outputs = net(input_tensor)
    
    # Process all three resolution levels
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create subdirectories for each resolution
    os.makedirs(os.path.join(output_dir, 'low_res'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mid_res'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'high_res'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'combined'), exist_ok=True)
    
    # Resolution names and scale factors for resizing the original image
    resolutions = [
        ('low_res', 0.25),   # 1/4 resolution
        ('mid_res', 0.5),    # 1/2 resolution
        ('high_res', 1.0)    # Full resolution
    ]
    
    # Lists to store outputs for combined visualization
    dx_list = []
    dy_list = []
    mag_list = []
    img_list = []
    res_names = []
    
    # Process each resolution level
    for i, (res_name, scale) in enumerate(resolutions):
        output = outputs[i]  # Get output at this resolution
        
        # Split channels
        dx = output[0].clamp(-0.5, 0.5)
        dy = output[1].clamp(-0.5, 0.5)
        mag = output[2].clamp(-0.5, 0.5)
        
        # Store for combined visualization
        dx_list.append(dx[0])
        dy_list.append(dy[0])
        mag_list.append(mag[0])
        res_names.append(res_name)
        
        # Create a resized version of the original image for visualization
        if scale != 1.0:
            # Resize the original image to match this resolution level
            width, height = original_img.size
            new_width, new_height = int(width * scale), int(height * scale)
            resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Store for combined visualization
            img_list.append(resized_img)
            
            # Temporarily replace the global original_img for visualization
            temp_img = original_img
            original_img = resized_img
        else:
            img_list.append(original_img)
        
        # Save visualization for this resolution
        vis_path = os.path.join(output_dir, res_name, f"{base_name}_blur_field.png")
        visualize_blur_field(dx[0], dy[0], mag[0], vis_path, 
                            title_prefix=f"{res_name.replace('_', ' ').title()}: ")
        
        # Save direction-color visualization
        dir_color_path = os.path.join(output_dir, res_name, f"{base_name}_direction_color.png")
        visualize_direction_color(dx[0], dy[0], mag[0], original_img, dir_color_path,
                                 title=f"{res_name.replace('_', ' ').title()}")
        
        # Restore original image if we changed it
        if scale != 1.0:
            original_img = temp_img
    
    # Create combined visualization of all resolutions
    combined_path = os.path.join(output_dir, 'combined', f"{base_name}_all_resolutions.png")
    visualize_all_resolutions(dx_list, dy_list, mag_list, img_list, res_names, combined_path)
    
    # Create blur conditions visualization (similar to reference image)
    conditions_path = os.path.join(output_dir, 'combined', f"{base_name}_blur_conditions.png")
    condition_names = ["Low Resolution", "Mid Resolution", "High Resolution"]
    visualize_blur_conditions(dx_list, dy_list, mag_list, conditions_path, condition_names)
    
    print(f"Inference completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, type=str, help="Path to input blurred image")
    parser.add_argument("--output_dir", default="./results", type=str, help="Output directory")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Device to use")
    args = parser.parse_args()
    
    test_blur_field(args.image_path, args.output_dir, args.device)
