import torch
import torch.nn as nn
import os
# from .dpt_lib.blocks import Interpolate # Original problematic import

# Try relative import first, then direct if run as script
try:
    from .dpt_lib.blocks import Interpolate
except ImportError:
    from dpt_lib.blocks import Interpolate

# Import DPT model from dpt_lib
try:
    from .dpt_lib.models import DPT
except ImportError:
    # Fallback if running model_utils.py directly for some reason,
    # or if the relative import path needs adjustment depending on execution context.
    try:
        from dpt_lib.models import DPT
    except ImportError:
        raise ImportError("Could not import DPT model from dpt_lib. Check paths.")


def _create_blur_head(head_type, features, output_channels):
    """Helper function to create different blur head architectures."""
    if head_type == "original_blur_head":
        print(f"Using 'original_blur_head' ({features=}, {output_channels=})")
        return nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            # nn.Dropout(0.1, False), # Optional dropout
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0, bias=True), # Added bias=True for consistency
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        )
    elif head_type == "lightweight_blur_head":
        print(f"Using 'lightweight_blur_head' ({features=}, {output_channels=})")
        # Lightweight: Conv(256,64,1) -> BN,ReLU -> Conv(64,32,3) -> BN,ReLU -> Conv(32,3,1) -> Interp
        return nn.Sequential(
            nn.Conv2d(features, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0, bias=True),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        )
    elif head_type == "medium_blur_head":
        print(f"Using 'medium_blur_head' ({features=}, {output_channels=})")
        # Medium: Conv(256,96,1) -> BN,ReLU -> Conv(96,48,3) -> BN,ReLU -> Conv(48,3,1) -> Interp
        return nn.Sequential(
            nn.Conv2d(features, 96, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            nn.Conv2d(48, output_channels, kernel_size=1, stride=1, padding=0, bias=True),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        )
    else:
        raise ValueError(f"Unsupported blur_head_type: {head_type}. Choose 'original_blur_head', 'lightweight_blur_head', or 'medium_blur_head'.")


def create_dpt_blur_model(output_channels=3, model_type="dpt_hybrid", blur_head_type="original_blur_head", 
                          pretrained_weights_path=None, freeze_backbone=True):
    """
    Creates the DPT model, loads pre-trained DPT backbone weights, and replaces the head
    with a specified blur prediction head.

    Args:
        output_channels (int): Number of output channels for the regression head.
        model_type (str): Type of DPT model, 'dpt_large' or 'dpt_hybrid'.
        blur_head_type (str): Type of blur head architecture to use.
                              Options: 'original_blur_head', 'lightweight_blur_head', 'medium_blur_head'.
        pretrained_weights_path (str, optional): Path to pre-trained DPT weights for the backbone.
        freeze_backbone (bool): If True, freezes the parameters of the DPT backbone.

    Returns:
        torch.nn.Module: The configured DPT model with the new blur head.
    """
    print(f"Creating DPT model (type: {model_type}) with blur head (type: {blur_head_type}) for {output_channels}-channel output.")

    # Determine backbone string and features based on model_type
    if model_type == "dpt_large":
        backbone = "vitl16_384"
        features = 256
    elif model_type == "dpt_hybrid":
        backbone = "vitb_rn50_384"
        features = 256
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'dpt_large' or 'dpt_hybrid'.")

    model = DPT(head=nn.Identity(), backbone=backbone, features=features, use_bn=True)

    if pretrained_weights_path:
        print(f"Loading pre-trained DPT BACKBONE weights from: {pretrained_weights_path}")
        if not os.path.exists(pretrained_weights_path):
             raise FileNotFoundError(f"Pretrained DPT backbone weights not found at {pretrained_weights_path}")
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("scratch.output_conv.") and not k.startswith("auxlayer."):
                filtered_state_dict[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        print("Pre-trained DPT backbone weights loaded.")
        if unexpected_keys:
             print(f"  Warning: Unexpected keys when loading backbone weights: {unexpected_keys}")
        critical_missing = [k for k in missing_keys if not k.startswith("scratch.output_conv.")]
        if critical_missing:
             print(f"  Warning: Missing non-head keys during backbone weight load: {critical_missing}")
    else:
        print("Warning: No pre-trained backbone weights path provided. DPT backbone will use its default initialization.")

    if freeze_backbone:
        print("Freezing DPT backbone parameters...")
        num_frozen = 0
        for name, param in model.named_parameters():
            if not name.startswith("scratch.output_conv."):
                param.requires_grad = False
                num_frozen += param.numel()
            else:
                 param.requires_grad = True
        print(f"Froze {num_frozen:,} parameters in the DPT backbone.")
    else:
        print("DPT backbone parameters will remain trainable.")

    # Replace the head using the helper function
    model.scratch.output_conv = _create_blur_head(blur_head_type, features, output_channels)
    
    print(f"Initialized new blur head (type: {blur_head_type}) for {output_channels}-channel regression.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in DPT model: {total_params:,}")
    print(f"Trainable parameters (head only if backbone frozen): {trainable_params:,}")
    if total_params > 0 :
        print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")
    else:
        print("Percentage of trainable parameters: N/A (total params is 0)")


    return model

if __name__ == '__main__':
    print("Testing model_utils.py...")
    
    # Common settings for tests
    output_channels_test = 3
    # Example path, replace with your actual path or set to None
    # For DPT Hybrid ADE20K: 'weights/dpt_hybrid-ade20k-53898607.pt'
    # For DPT Large ADE20K: 'weights/dpt_large-ade20k-b12dca68.pt'
    hybrid_weights_path = 'weights/dpt_hybrid-ade20k-53898607.pt' 
    if not os.path.exists(hybrid_weights_path):
        print(f"Warning: Test weights {hybrid_weights_path} not found. Backbone will use default init for hybrid test.")
        hybrid_weights_path = None

    try:
        # Test DPT Hybrid with different head types
        print("\n--- Testing DPT Hybrid with Original Blur Head ---")
        model_hybrid_orig = create_dpt_blur_model(
            model_type="dpt_hybrid", 
            blur_head_type="original_blur_head",
            pretrained_weights_path=hybrid_weights_path, 
            freeze_backbone=True, 
            output_channels=output_channels_test
        )
        dummy_input_hybrid = torch.randn(1, 3, 384, 384)
        output_hybrid_orig = model_hybrid_orig(dummy_input_hybrid)
        print(f"Hybrid model (original head) output shape: {output_hybrid_orig.shape}")

        print("\n--- Testing DPT Hybrid with Lightweight Blur Head ---")
        model_hybrid_light = create_dpt_blur_model(
            model_type="dpt_hybrid", 
            blur_head_type="lightweight_blur_head",
            pretrained_weights_path=hybrid_weights_path,
            freeze_backbone=True, 
            output_channels=output_channels_test
        )
        output_hybrid_light = model_hybrid_light(dummy_input_hybrid)
        print(f"Hybrid model (lightweight head) output shape: {output_hybrid_light.shape}")

        print("\n--- Testing DPT Hybrid with Medium Blur Head ---")
        model_hybrid_medium = create_dpt_blur_model(
            model_type="dpt_hybrid",
            blur_head_type="medium_blur_head",
            pretrained_weights_path=hybrid_weights_path,
            freeze_backbone=True, 
            output_channels=output_channels_test
        )
        output_hybrid_medium = model_hybrid_medium(dummy_input_hybrid)
        print(f"Hybrid model (medium head) output shape: {output_hybrid_medium.shape}")

        # Example for DPT Large (if you have weights and want to test)
        # large_weights_path = 'weights/dpt_large-ade20k-b12dca68.pt'
        # if not os.path.exists(large_weights_path):
        #     print(f"Warning: Test weights {large_weights_path} not found. Skipping DPT Large test or it will use default init.")
        #     large_weights_path = None
        #
        # if large_weights_path: # Only run if weights are found or you want default init
        #     print("\n--- Testing DPT Large with Original Blur Head ---")
        #     model_large = create_dpt_blur_model(
        #         model_type="dpt_large", 
        #         blur_head_type="original_blur_head", # Or other head types
        #         pretrained_weights_path=large_weights_path, 
        #         freeze_backbone=True, 
        #         output_channels=output_channels_test)
        #     dummy_input_large = torch.randn(1, 3, 384, 384)
        #     output_large = model_large(dummy_input_large)
        #     print(f"Large model output shape: {output_large.shape}")

        print("\nModel creation test successful.")
    except Exception as e:
        print(f"Error during model_utils.py test: {e}")
        raise 