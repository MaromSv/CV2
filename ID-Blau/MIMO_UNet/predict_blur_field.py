import torch
import os
import sys
import argparse
from tqdm import tqdm

# Add parent directory to path to access DPT_blur
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import directly from DPT_blur
from DPT_blur.visualize_blur_map import visualize_blur_map
from DPT_blur.data_loader import BlurMapDataset
from DPT_blur.model_utils import create_dpt_blur_model

def predict_blur_field(model, dataset, save_dir, device, factor=8):
    """Run inference with MIMO-UNet model for blur field prediction"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            input_img = sample['blur'].unsqueeze(0).to(device)
            
            # Handle padding similar to other deblur models
            b, c, h, w = input_img.shape
            h_n = (factor - h % factor) % factor
            w_n = (factor - w % factor) % factor
            input_img = torch.nn.functional.pad(input_img, (0, w_n, 0, h_n), mode='reflect')
            
            # Get model outputs
            outputs = model(input_img)
            
            # Extract blur field components (dx, dy, magnitude)
            blur_field = outputs[:, :, :h, :w]
            
            # Save the blur field tensor
            image_name = os.path.split(dataset.get_path(idx=idx)['blur_path'])[-1]
            save_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_blur_field.pt")
            torch.save(blur_field[0].cpu(), save_path)
            
            # Optionally visualize using DPT_blur's visualization function
            if idx < 5:  # Visualize first few samples
                visualize_blur_map(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict blur fields using MIMO-UNet")
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--save_dir', type=str, default='results/blur_fields', help='Directory to save results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset using DPT_blur's dataset class
    dataset = BlurMapDataset(
        blurred_dir=args.data_path,
        gt_dir=None,  # No ground truth needed for inference
        transform=None,  # Add appropriate transforms if needed
        is_train=False
    )
    
    # Load MIMO-UNet model
    # (Your MIMO-UNet model loading code here)
    
    # Run prediction
    predict_blur_field(model, dataset, args.save_dir, device)