import torch
import torch.nn as nn
import os

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


def create_dpt_blur_model(output_channels=3, model_type="dpt_hybrid", pretrained_weights_path=None, freeze_backbone=True):
    """
    Creates the DPT model, loads pre-trained DPT backbone weights, and replaces the head.

    Args:
        output_channels (int): Number of output channels for the regression head (e.g., 3 for bx, by, magnitude).
        model_type (str): Type of DPT model, 'dpt_large' or 'dpt_hybrid'.
        pretrained_weights_path (str, optional): Path to pre-trained DPT weights for the backbone.
                                                 If None, backbone is initialized with its default pre-training
                                                 (usually ImageNet if not specified otherwise by the DPT library).
        freeze_backbone (bool): If True, freezes the parameters of the DPT backbone.

    Returns:
        torch.nn.Module: The configured DPT model with a new regression head.
    """
    print(f"Creating DPT model (type: {model_type}) for {output_channels}-channel output.")

    # Determine backbone string and features based on model_type
    if model_type == "dpt_large":
        # DPT Large (ViT-L/16)
        backbone = "vitl16_384"
        features = 256 # Standard feature size for DPT head
    elif model_type == "dpt_hybrid":
        # DPT Hybrid (ViT-B/16 with ResNet50 encoder)
        backbone = "vitb_rn50_384"
        features = 256 # Standard feature size for DPT head
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'dpt_large' or 'dpt_hybrid'.")

    # Create base DPT model.
    # The head provided here is just a placeholder; it will be replaced.
    # `use_bn=True` is often good for segmentation/regression heads.
    model = DPT(head=nn.Identity(), backbone=backbone, features=features, use_bn=True)

    # Load pre-trained weights for the BACKBONE if path is provided
    if pretrained_weights_path:
        print(f"Loading pre-trained DPT BACKBONE weights from: {pretrained_weights_path}")
        if not os.path.exists(pretrained_weights_path):
             raise FileNotFoundError(f"Pretrained DPT backbone weights not found at {pretrained_weights_path}")

        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")

        # The DPT library might store weights directly or under a 'state_dict' key.
        # It might also have "module." prefix if saved from DataParallel.
        if "state_dict" in checkpoint:
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}


        # Filter out keys related to the original head (e.g., segmentation head, aux layer)
        # to load only the backbone weights.
        # DPT's head is typically in `scratch.output_conv` and an aux layer might exist.
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("scratch.output_conv.") and not k.startswith("auxlayer."):
                filtered_state_dict[k] = v
            # else:
            #     print(f"  DEBUG: Excluding key from pretrained: {k}") # Optional debug

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

        print("Pre-trained DPT backbone weights loaded.")
        if unexpected_keys:
             print(f"  Warning: Unexpected keys when loading backbone weights: {unexpected_keys}")
        # We expect missing keys for `scratch.output_conv` as we are replacing it.
        # Check if any other crucial parts are missing.
        critical_missing = [k for k in missing_keys if not k.startswith("scratch.output_conv.")]
        if critical_missing:
             print(f"  Warning: Missing non-head keys during backbone weight load: {critical_missing}")
    else:
        print("Warning: No pre-trained backbone weights path provided. DPT backbone will use its default initialization.")

    # Freeze backbone parameters (if requested)
    if freeze_backbone:
        print("Freezing DPT backbone parameters...")
        num_frozen = 0
        for name, param in model.named_parameters():
            # Keep the head (output_conv) unfrozen, freeze everything else.
            if not name.startswith("scratch.output_conv."):
                param.requires_grad = False
                num_frozen += 1
            else:
                 param.requires_grad = True # Ensure new head is trainable
        print(f"Froze {num_frozen} parameters in the DPT backbone.")
    else:
        print("DPT backbone parameters will remain trainable.")


    # Replace the head with a new regression head for blur map (bx, by, magnitude)
    # This is an example head; adjust architecture as needed.
    model.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            # nn.Dropout(0.1, False), # Optional dropout
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0) # Output bx, by, magnitude
            # No final activation like Sigmoid/Tanh unless your GT is normalized to a specific range
        )
    print(f"Initialized new model head for {output_channels}-channel regression.")

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in DPT model: {total_params:,}")
    print(f"Trainable parameters (mostly head): {trainable_params:,}")

    return model

if __name__ == '__main__':
    # Example usage:
    print("Testing model_utils.py...")
    # Create a dummy model (no pre-trained weights for this test)
    try:
        # Test Hybrid
        print("\n--- Testing DPT Hybrid ---")
        model_hybrid = create_dpt_blur_model(model_type="dpt_hybrid", pretrained_weights_path=None, freeze_backbone=True, output_channels=3)
        dummy_input_hybrid = torch.randn(1, 3, 384, 384)
        output_hybrid = model_hybrid(dummy_input_hybrid)
        print(f"Hybrid model output shape: {output_hybrid.shape}")

        # Test Large
        print("\n--- Testing DPT Large ---")
        model_large = create_dpt_blur_model(model_type="dpt_large", pretrained_weights_path=None, freeze_backbone=False, output_channels=3)
        dummy_input_large = torch.randn(1, 3, 384, 384)
        output_large = model_large(dummy_input_large)
        print(f"Large model output shape: {output_large.shape}")

        print("\nModel creation test successful.")
    except Exception as e:
        print(f"Error during model_utils.py test: {e}")
        raise 