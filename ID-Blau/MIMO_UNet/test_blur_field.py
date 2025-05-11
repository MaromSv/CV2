import torch
import torch.nn as nn
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from MIMOUNet import build_MIMOUnet_net

def visualize_blur_field(dx, dy, mag, save_path):
    """Create a visualization of the blur field using arrows"""
    # Convert tensors to numpy arrays
    dx_np = dx.squeeze().cpu().numpy()
    dy_np = dy.squeeze().cpu().numpy()
    mag_np = mag.squeeze().cpu().numpy()
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot original image
    axs[0, 0].imshow(np.array(original_img))
    axs[0, 0].set_title('Original Blurred Image')
    axs[0, 0].axis('off')
    
    # Plot magnitude as heatmap
    mag_plot = axs[0, 1].imshow(mag_np, cmap='viridis')
    axs[0, 1].set_title('Blur Magnitude')
    axs[0, 1].axis('off')
    plt.colorbar(mag_plot, ax=axs[0, 1])
    
    # Plot dx
    dx_plot = axs[1, 0].imshow(dx_np, cmap='coolwarm')
    axs[1, 0].set_title('Displacement X')
    axs[1, 0].axis('off')
    plt.colorbar(dx_plot, ax=axs[1, 0])
    
    # Plot dy
    dy_plot = axs[1, 1].imshow(dy_np, cmap='coolwarm')
    axs[1, 1].set_title('Displacement Y')
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
    
    plt.title('Blur Field Direction Visualization')
    plt.axis('off')
    plt.tight_layout()
    vector_path = save_path.replace('.png', '_vectors.png')
    plt.savefig(vector_path, dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {save_path} and {vector_path}")

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
    
    # Debug: Print the output structure
    print(f"Output type: {type(outputs)}")
    print(f"Output length: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output[{i}] type: {type(out)}")
        print(f"Output[{i}] shape: {out.shape if hasattr(out, 'shape') else 'No shape'}")
    
    # Assuming the model outputs a list of tensors, and we need to split the channels
    # Get full resolution output (last in the list)
    output = outputs[2]  # This should be a tensor with 3 channels
    
    # Check if we need to split channels
    if output.shape[1] == 3:  # If it has 3 channels
        # Split the channels
        dx = output[:, 0:1, :, :]
        dy = output[:, 1:2, :, :]
        mag = output[:, 2:3, :, :]
    else:
        # If it's not what we expected, just use the same tensor for all three
        # This is just for debugging
        print(f"Warning: Expected 3 channels but got {output.shape[1]}. Using the same output for all visualizations.")
        dx = output
        dy = output
        mag = output
    
    # Crop to original size
    dx = dx[:, :, :h, :w].clamp(-0.5, 0.5)
    dy = dy[:, :, :h, :w].clamp(-0.5, 0.5)
    mag = mag[:, :, :h, :w].clamp(-0.5, 0.5)
    
    # Save visualization
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(output_dir, f"{base_name}_blur_field.png")
    visualize_blur_field(dx[0, 0], dy[0, 0], mag[0, 0], vis_path)
    
    print(f"Inference completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, type=str, help="Path to input blurred image")
    parser.add_argument("--output_dir", default="./results", type=str, help="Output directory")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Device to use")
    args = parser.parse_args()
    
    test_blur_field(args.image_path, args.output_dir, args.device)
