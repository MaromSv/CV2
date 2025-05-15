import os
import sys
import torch
import yaml
import argparse
from PIL import Image
from torchvision import transforms

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import from MIMO-UNet
from MIMOUNet import build_MIMOUnet_net

# Import from DPT_blur
from DPT_blur.visualize_blur_map import visualize_blur_map

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@torch.no_grad()
def test_blur_field(image_path, output_dir, config, device='cuda'):
    """Test blur field prediction on a single image"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_name = config['model']['name']
    print(f"Initializing {model_name} with random weights...")
    net = build_MIMOUnet_net(model_name)
    
    # Try to load weights if they exist, but don't require them
    model_weights = config['paths'].get('model_weights', '')
    if model_weights and os.path.exists(model_weights):
        print(f"Loading weights from {model_weights}")
        state_dict = torch.load(model_weights, map_location=device)
        # Remove 'module.' prefix if model was saved with DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
    else:
        print("Using random initialization (no pre-trained weights)")
    
    net = net.to(device)
    net.eval()
    
    # Load and preprocess image
    print(f"Processing image: {image_path}")
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
    print("Running model inference...")
    outputs = net(input_tensor)
    
    # Process output (assuming the model returns a list of outputs at different resolutions)
    # We use the highest resolution output (last in the list)
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[-1], tuple) and len(outputs[-1]) == 3:
            # If output is (dx, dy, mag) tuple
            dx, dy, mag = outputs[-1]
            # Combine into a single tensor
            blur_field = torch.cat([dx, dy, mag], dim=1)
        else:
            # If output is already a tensor
            blur_field = outputs[-1]
    else:
        # If output is already a tensor
        blur_field = outputs
    
    # Crop to original size
    blur_field = blur_field[:, :, :h, :w]
    
    # Save tensor
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    tensor_path = os.path.join(output_dir, f"{base_name}_blur_field.pt")
    torch.save(blur_field[0].cpu(), tensor_path)
    
    print(f"Saved blur field tensor to: {tensor_path}")
    print("Generating visualization...")
    
    # Visualize using DPT_blur's visualization function
    quiver_step = config['visualization'].get('quiver_step', 16)
    vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    visualize_blur_map(tensor_path, image_path, quiver_step, output_path=vis_path)
    
    print(f"Saved visualization to: {vis_path}")
    print("Done!")
    
    return tensor_path, vis_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MIMO-UNet blur field prediction on a single image")
    parser.add_argument("--image_path", type=str, help="Path to input blurred image")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use (overrides config)")
    args = parser.parse_args()
    
    # Create default config if it doesn't exist
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Creating default config.")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_config = {
            "paths": {
                "model_weights": "",  # Empty by default
                "test_data": "./dataset/test",
                "results_dir": "./results"
            },
            "model": {
                "name": "MIMO-UNetPlus",
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "visualization": {
                "quiver_step": 16,
                "save_visualizations": True
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    # Load configuration
    config = load_config(config_path)
    
    # Override config with command line arguments if provided
    device = args.device if args.device else config['model'].get('device', 'cuda')
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    output_dir = args.output_dir if args.output_dir else config['paths'].get('results_dir', './results')
    
    # If image_path is not provided, use a default test image
    image_path = args.image_path
    if not image_path:
        test_data_dir = config['paths'].get('test_data', './dataset/test')
        # Try to find an image in the test directory
        if os.path.exists(test_data_dir):
            for ext in ['.png', '.jpg', '.jpeg']:
                test_images = [f for f in os.listdir(test_data_dir) if f.endswith(ext)]
                if test_images:
                    image_path = os.path.join(test_data_dir, test_images[0])
                    break
        
        if not image_path:
            # Use a sample from GOPRO dataset if available
            gopro_path = "../dataset/GOPRO_Large/test/GOPR0384_11_00/blur/000001.png"
            if os.path.exists(gopro_path):
                image_path = gopro_path
            else:
                print("No test image found. Please provide an image path.")
                sys.exit(1)
    
    # Run test
    test_blur_field(image_path, output_dir, config, device)
