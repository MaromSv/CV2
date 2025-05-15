import os
import sys
import torch
import yaml

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

# Import directly from DPT_blur
from DPT_blur.visualize_blur_map import visualize_blur_map
from DPT_blur.test_inference import run_dpt_blur_prediction

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_mimo_blur_map(tensor_path, image_path=None, quiver_step=16, output_path=None):
    """Wrapper around DPT_blur's visualization function"""
    return visualize_blur_map(tensor_path, image_path, quiver_step, output_path)

def process_blur_prediction(tensor, model, device):
    """Process a tensor through the model and return blur field prediction"""
    model.eval()
    with torch.no_grad():
        # Handle padding if needed
        b, c, h, w = tensor.shape
        factor = 8
        h_n = (factor - h % factor) % factor
        w_n = (factor - w % factor) % factor
        padded_tensor = torch.nn.functional.pad(tensor, (0, w_n, 0, h_n), mode='reflect')
        
        # Forward pass
        outputs = model(padded_tensor.to(device))
        
        # Process output based on model type
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
        
        return blur_field

def save_blur_field(blur_field, output_path):
    """Save blur field tensor to disk"""
    torch.save(blur_field.cpu(), output_path)
    return output_path
