import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import sys
from visualize_blur_map import create_color_wheel

# Add parent directory to path to import dataset
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def visualize_multiple_blur_fields(tensor_list, image_path_list=None, output_dir="./test_visualizations"):
    """Visualize multiple blur fields with a shared color wheel legend"""
    os.makedirs(output_dir, exist_ok=True)
    
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
    ax_legend.text(orient_x + 45, orient_y - 45, "Orientation", ha='center', va='center', 
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
        import matplotlib.colors as mcolors
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
    output_path = os.path.join(output_dir, "multiple_blur_fields.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path

def test_with_dataset_samples(dataset_path, output_dir="./test_visualizations", num_samples=5):
    """Test visualization with real samples from the dataset"""
    import glob
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paths to blur images and condition files
    blur_dir = os.path.join(dataset_path, 'blur')
    condition_dir = os.path.join(dataset_path, 'condition')
    
    if not os.path.exists(blur_dir):
        print(f"Error: Blur directory not found at {blur_dir}")
        return
    
    if not os.path.exists(condition_dir):
        print(f"Error: Condition directory not found at {condition_dir}")
        return
    
    # Get all blur images
    blur_images = sorted(glob.glob(os.path.join(blur_dir, '*.png')))
    if not blur_images:
        blur_images = sorted(glob.glob(os.path.join(blur_dir, '*.jpg')))
    
    if not blur_images:
        print(f"No image files found in {blur_dir}")
        return
    
    # Get all condition files
    condition_files = sorted(glob.glob(os.path.join(condition_dir, '*.npy')))
    
    if not condition_files:
        print(f"No .npy files found in {condition_dir}")
        return
    
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
            
            # Create individual visualization for this sample using our new function
            try:
                # Import the new visualization function
                from visualize_blur_map import visualize_blur_components
                
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
                from visualize_blur_map import visualize_blur_field_with_legend
                
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
        return
    
    # Create multiple visualization
    try:
        visualize_multiple_blur_fields(tensor_list, image_path_list, output_dir)
        print("Multiple visualization created successfully")
    except Exception as e:
        print(f"Error creating multiple visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize blur fields from dataset")
    parser.add_argument("--dataset_path", type=str, default="../ID-Blau/dataset/small_dataset/train", 
                        help="Path to dataset directory containing 'blur' and 'condition' folders")
    parser.add_argument("--output_dir", default="./test_visualizations", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    # Print current directory and sys.path for debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    
    test_with_dataset_samples(args.dataset_path, args.output_dir, args.num_samples)
