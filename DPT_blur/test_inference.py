import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from dpt_lib.models import DPT
from dpt_lib.transforms import Resize, NormalizeImage, PrepareForNet
from dpt_lib.blocks import Interpolate
import cv2
import os

def run_dpt_blur_prediction(tensor, model_path, output_channels=3, model_type="dpt_hybrid", optimize=True, print_shapes=False):
    """Run DPT model adapted for blur vector (bx, by) and magnitude prediction on a tensor.

    Loads a pre-trained DPT model, replaces its head
    for regression, and predicts a 3-channel map representing the
    X and Y components of the blur vector and the blur magnitude at each pixel.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        model_path (str): Path to the pre-trained DPT model weights.
        output_channels (int): Number of output channels (MUST be 3 for bx, by, magnitude). Defaults to 3.
        model_type (str): DPT model type ("dpt_large" or "dpt_hybrid"). Defaults to "dpt_hybrid".
        optimize (bool): Whether to use optimization for CUDA/MPS. Defaults to True.
        print_shapes (bool): If True, print shapes during inference. Defaults to False.

    Returns:
        torch.Tensor: Blur predictions (bx, by, magnitude) of shape (B, 3, H, W).
    """
    if output_channels != 3:
        raise ValueError("output_channels must be 3 for (bx, by, magnitude) vector prediction.")

    # --- Device Selection --- 
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Determine backbone and features based on model_type
    if model_type == "dpt_large":
        backbone = "vitl16_384"
        features = 256 # DPT Large uses 256 features in the head
    elif model_type == "dpt_hybrid":
        backbone = "vitb_rn50_384"
        features = 256 # DPT Hybrid uses 256 features in the head
    else:
        raise ValueError(f"model_type '{model_type}' not implemented, use: dpt_large or dpt_hybrid")

    # Load the base DPT model structure
    print("Creating DPT model...")
    model = DPT(head=nn.Identity(), backbone=backbone, features=features, use_bn=True)

    # Load the pre-trained weights
    print(f"Loading weights from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint

    # Filter out head/aux weights if they exist in the checkpoint
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("scratch.output_conv.") and not k.startswith("auxlayer."):
             filtered_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    print("Loaded pre-trained DPT weights.")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
    if missing_keys and any(not (k.startswith("scratch.output_conv.") or k.startswith("auxlayer.")) for k in missing_keys):
        print(f"Warning: Missing non-head keys: {[k for k in missing_keys if not (k.startswith('scratch.output_conv.') or k.startswith('auxlayer.'))]}")


    # Replace the head with a new regression head
    model.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1), # Mimic first part of original depth head
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0), # Final conv to output_channels
        )
    print(f"Replaced model head for {output_channels}-channel regression.")

    # --- Model Summary / Layer Shapes (Optional) ---
    if print_shapes:
        print("\n--- Model Architecture ---")
        print(model)
        # Note: For detailed input/output shapes per layer, consider torchinfo or torchsummary
        # Or print shapes during the forward pass (less clean)
        print("------------------------\n")


    # Set up transforms
    transform = Compose([
        Resize(
            384, 384, # Use the backbone's native resolution
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])

    model.eval()

    # Optimization for CUDA/MPS (Channels Last)
    # Note: Channels Last might not be fully supported or optimal on MPS/CPU
    if optimize and (device == torch.device("cuda") or device == torch.device("mps")):
        try:
            model = model.to(memory_format=torch.channels_last)
            print("Using memory format: channels_last")
        except Exception as e:
            print(f"Warning: Could not set memory format to channels_last on {device}: {e}")
        # FP16 (Half precision) - Use with caution for regression & non-CUDA devices
        # if device == torch.device("cuda"): 
        #    model = model.half()
        #    print("Using precision: FP16 (CUDA only)")

    model.to(device)

    # Process the tensor
    # The input `tensor` is expected to be preprocessed and in the correct shape (B, C, H, W)
    # If it's coming directly from an image file, preprocessing would happen before calling this function.
    input_tensor = tensor.to(device)
    original_shape = input_tensor.shape[2:] # Store H, W of the input tensor

    with torch.no_grad():
        # The DPT transforms (Resize, NormalizeImage, PrepareForNet) are typically applied
        # to the input *before* it becomes the `tensor` argument to this function.
        # Here, we assume `tensor` is already the output of `PrepareForNet`.
        sample = input_tensor # Already a tensor (B, C, H, W)

        if print_shapes: print(f"Input tensor shape: {sample.shape}")

        if optimize and (device == torch.device("cuda") or device == torch.device("mps")):
            # Check if model attributes channels_last to avoid errors if format conversion failed
            if hasattr(model, 'channels_last') and model.channels_last:
                 sample = sample.to(memory_format=torch.channels_last)
            # if device == torch.device("cuda"): 
            #    sample = sample.half()

        # --- DPT Forward Pass --- 
        if print_shapes: print("Running model.forward()...")
        raw_output = model.forward(sample) # Shape (B, output_channels, H_feat, W_feat)
        if print_shapes: print(f"  Model output (raw, before interpolate): {raw_output.shape}")
        # ------------------------

        # Resize back to original input tensor's size (which might be different from original image if resized by transform)
        prediction = F.interpolate(
            raw_output,
            size=original_shape,  # Resize to original H, W of the *input tensor*
            mode="bilinear",
            align_corners=False
        )
        if print_shapes: print(f"Final Prediction shape: {prediction.shape}")

    return prediction

if __name__ == "__main__":
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser(description="Run DPT model for blur quantification on an image.")
    parser.add_argument('--image_path', type=str, default='data/GOPRO_Large/test/GOPR0384_11_00/blur/000001.png', help='Path to the input image file.')
    parser.add_argument('--weights', type=str, default="weights/dpt_large-ade20k-b12dca68.pt", help='Path to pre-trained DPT model weights (.pt file).')
    parser.add_argument('--model_type', type=str, default='dpt_large', choices=['dpt_hybrid', 'dpt_large'], help='DPT model type.')
    parser.add_argument('--output_file', type=str, default='results/output_blur_map.pt', help='Path to save the output blur map tensor.')
    parser.add_argument('--print_shapes', action='store_true', help='Print model layer shapes during inference.')
    parser.add_argument('--no_optimize', action='store_true', help='Disable optimizations (memory format).')
    args = parser.parse_args()

    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        exit()
    else:
        print(f"Loading image from {args.image_path}")

    # Load and preprocess the image
    try:
        img_np = cv2.imread(args.image_path) # BGR, HWC, uint8
        if img_np is None:
            raise IOError(f"Failed to read image from {args.image_path}")
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # RGB, HWC, uint8
        img_np = img_np.astype(np.float32) # RGB, HWC, float32
    except Exception as e:
        print(f"Error loading or processing image {args.image_path}: {e}")
        exit()

    # Define the DPT transforms (same as used in training and visualization_helpers if any)
    # Note: The target size 384, 384 might need to match what the model expects or was trained on.
    # If the model has a fixed input size, this resize is crucial.
    # For DPT, common input sizes are 384x384 or 512x512.
    net_w, net_h = 384, 384 # Example, adjust if your DPT model expects different input size
    if args.model_type == "dpt_large": # Typically 384x384 or 512x512 for ViT-L
        net_w, net_h = 384, 384
    elif args.model_type == "dpt_hybrid": # Typically 384x384 for ViT-B based hybrids
        net_w, net_h = 384, 384

    dpt_transform = Compose([
        Resize(
            net_w, net_h,
            resize_target=None, # Resize the image itself
            keep_aspect_ratio=True,
            ensure_multiple_of=32, # DPT requirement
            resize_method="minimal", # Method might be less critical if keep_aspect_ratio is False
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Standard DPT normalization
        PrepareForNet(), # Converts to CHW tensor and permutes
    ])

    # Apply transforms to the loaded image
    transformed_sample = dpt_transform({"image": img_np}) # Expects dict, returns dict
    input_tensor = torch.from_numpy(transformed_sample["image"]).unsqueeze(0) # Add batch dimension (B, C, H, W)
    print(f"Processed input tensor of shape: {input_tensor.shape}")

    # Check if model weights file exists
    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found at {args.weights}")
        print("Please download the weights or provide the correct path.")
        exit()
    else:
        print(f"Found model weights at {args.weights}")

    # Run blur prediction
    blur_predictions = run_dpt_blur_prediction(
        input_tensor, # Use the processed image tensor
        model_path=args.weights,
        output_channels=3, # Must be 3 for (bx, by, magnitude)
        model_type=args.model_type,
        optimize=not args.no_optimize,
        print_shapes=args.print_shapes
    )

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the output tensor (only the first item in the batch)
    output_tensor_to_save = blur_predictions[0].cpu() # Move to CPU before saving
    try:
        torch.save(output_tensor_to_save, args.output_file)
        print(f"Output blur map saved to: {args.output_file}")
        print(f"Saved tensor shape: {output_tensor_to_save.shape}") # Should be (3, H_processed, W_processed)
    except Exception as e:
        print(f"Error saving output tensor to {args.output_file}: {e}") 