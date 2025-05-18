import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurFieldLoss(nn.Module):
    """
    Combined loss function for blur field prediction:
    - L1 loss for overall vector field accuracy
    - Cosine similarity loss for direction accuracy
    - Magnitude loss for blur strength accuracy
    """
    def __init__(self, lambda_dir=0.5, lambda_mag=0.5):
        super(BlurFieldLoss, self).__init__()
        self.lambda_dir = lambda_dir
        self.lambda_mag = lambda_mag
        self.charbonnier = CharbonnierLoss()
        
    def forward(self, pred, target):
        # Extract components
        pred_bx, pred_by = pred[:, 0:1, :, :], pred[:, 1:2, :, :]
        pred_mag = pred[:, 2:3, :, :]
        
        target_bx, target_by = target[:, 0:1, :, :], target[:, 1:2, :, :]
        target_mag = target[:, 2:3, :, :]
        
        # L1 loss on the entire field
        l1_loss = self.charbonnier(pred, target)
        
        # Directional loss (cosine similarity)
        pred_vectors = torch.cat([pred_bx, pred_by], dim=1)
        target_vectors = torch.cat([target_bx, target_by], dim=1)
        
        # Normalize vectors for cosine similarity
        pred_norm = torch.norm(pred_vectors, p=2, dim=1, keepdim=True) + 1e-8
        target_norm = torch.norm(target_vectors, p=2, dim=1, keepdim=True) + 1e-8
        
        pred_normalized = pred_vectors / pred_norm
        target_normalized = target_vectors / target_norm
        
        # Compute cosine similarity
        cos_sim = (pred_normalized * target_normalized).sum(dim=1, keepdim=True)
        dir_loss = (1 - cos_sim).mean()
        
        # Magnitude loss
        mag_loss = self.charbonnier(pred_mag, target_mag)
        
        # Combined loss
        total_loss = l1_loss + self.lambda_dir * dir_loss + self.lambda_mag * mag_loss
        
        return total_loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class MultiScaleBlurFieldLoss(nn.Module):
    """
    Multi-scale loss function for MIMO-UNet outputs.
    Computes loss at multiple scales and combines them with adaptive weights.
    """
    def __init__(self, base_criterion=None, scale_weights=None, use_consistency=True, consistency_weight=0.1):
        super(MultiScaleBlurFieldLoss, self).__init__()
        
        # Use BlurFieldLoss as default criterion if none provided
        self.base_criterion = base_criterion if base_criterion is not None else BlurFieldLoss()
        
        # Default scale weights (can be overridden during training)
        self.scale_weights = scale_weights if scale_weights is not None else [0.25, 0.5, 1.0]
        
        # Consistency loss between scales
        self.use_consistency = use_consistency
        self.consistency_weight = consistency_weight
        self.charbonnier = CharbonnierLoss()
        
    def forward(self, outputs, target, scale_weights=None):
        """
        Compute multi-scale loss for MIMO-UNet outputs.
        
        Args:
            outputs: List of model outputs at different scales or single output
            target: Ground truth blur field tensor [B, 3, H, W]
            scale_weights: Optional custom weights for this forward pass
        
        Returns:
            total_loss: Weighted sum of losses at all scales
            losses_dict: Dictionary of individual losses for logging
        """
        # Use provided weights or default weights
        weights = scale_weights if scale_weights is not None else self.scale_weights
        
        # Initialize total loss and losses dictionary
        total_loss = 0.0
        losses_dict = {}
        
        # Handle case where outputs is not a list (single output)
        if not isinstance(outputs, list):
            outputs = [outputs]
            print(f"DEBUG: Single output converted to list, shape: {outputs[0].shape}")
        else:
            print(f"DEBUG: Multi-scale outputs received, count: {len(outputs)}")
            for i, out in enumerate(outputs):
                if isinstance(out, tuple):
                    print(f"DEBUG: Scale {i} - tuple output with shapes: {out[0].shape}, {out[1].shape}, {out[2].shape}")
                else:
                    print(f"DEBUG: Scale {i} - tensor output with shape: {out.shape}")
        
        # Process each output scale
        for i, output in enumerate(outputs):
            if i >= len(weights):
                print(f"DEBUG: Skipping scale {i} as no weight is defined (weights length: {len(weights)})")
                break  # Skip if we don't have weights for this scale
            
            # Calculate scale factor for this level
            scale_factor = 0.25 * (2**i)  # 0.25, 0.5, 1.0
            print(f"DEBUG: Processing scale {i} with scale_factor {scale_factor}, weight {weights[i]}")
            
            # Create downsampled target for this scale
            if scale_factor < 1.0:
                # Use bilinear interpolation for continuous fields
                scaled_target = F.interpolate(target, scale_factor=scale_factor, 
                                             mode='bilinear', align_corners=False)
                print(f"DEBUG: Downsampled target from {target.shape} to {scaled_target.shape}")
            else:
                scaled_target = target
                print(f"DEBUG: Using original target shape {target.shape}")
            
            # Handle output format (tuple or tensor)
            if isinstance(output, tuple) and len(output) == 3:
                # If output is (dx, dy, mag) tuple
                dx, dy, mag = output
                # Combine into a single tensor
                pred = torch.cat([dx, dy, mag], dim=1)
                print(f"DEBUG: Combined tuple components into tensor of shape {pred.shape}")
            else:
                # If output is already a tensor
                pred = output
                print(f"DEBUG: Using tensor output directly, shape {pred.shape}")
            
            # Compute loss for this scale
            scale_loss = self.base_criterion(pred, scaled_target)
            print(f"DEBUG: Scale {i} loss: {scale_loss.item():.6f}")
            
            # Apply weight to this scale's loss
            weighted_loss = weights[i] * scale_loss
            total_loss += weighted_loss
            print(f"DEBUG: Scale {i} weighted loss ({weights[i]} * {scale_loss.item():.6f} = {weighted_loss.item():.6f})")
            
            # Store individual losses for logging
            scale_name = ['low', 'medium', 'high'][i] if i < 3 else f'scale_{i}'
            losses_dict[f'loss_{scale_name}'] = scale_loss.item()
        
        # Add consistency loss if enabled
        if self.use_consistency:
            consistency_loss = self._compute_consistency_loss(outputs)
            total_loss += self.consistency_weight * consistency_loss
            losses_dict['loss_consistency'] = consistency_loss.item()
        
        # Store total loss
        losses_dict['loss_total'] = total_loss.item()
        
        return total_loss, losses_dict
    
    def _compute_consistency_loss(self, outputs):
        """
        Compute consistency loss between different scales
        """
        consistency_loss = 0.0
        
        # Compare upsampled lower resolution to higher resolution
        for i in range(len(outputs) - 1):
            lower_res = outputs[i]
            higher_res = outputs[i + 1]
            
            # Handle output format (tuple or tensor)
            if isinstance(lower_res, tuple) and len(lower_res) == 3:
                # If output is (dx, dy, mag) tuple
                dx_lr, dy_lr, mag_lr = lower_res
                dx_hr, dy_hr, mag_hr = higher_res
                
                # Upsample each component
                dx_lr_up = F.interpolate(dx_lr, size=dx_hr.shape[2:], mode='bilinear', align_corners=False)
                dy_lr_up = F.interpolate(dy_lr, size=dy_hr.shape[2:], mode='bilinear', align_corners=False)
                mag_lr_up = F.interpolate(mag_lr, size=mag_hr.shape[2:], mode='bilinear', align_corners=False)
                
                # Compute consistency loss for each component
                dx_cons = self.charbonnier(dx_lr_up, dx_hr)
                dy_cons = self.charbonnier(dy_lr_up, dy_hr)
                mag_cons = self.charbonnier(mag_lr_up, mag_hr)
                
                consistency_loss += dx_cons + dy_cons + mag_cons
            else:
                # If outputs are tensors
                lower_res_up = F.interpolate(lower_res, size=higher_res.shape[2:], 
                                           mode='bilinear', align_corners=False)
                consistency_loss += self.charbonnier(lower_res_up, higher_res)
        
        return consistency_loss
    
    def update_scale_weights(self, new_weights):
        """
        Update scale weights during training
        """
        assert len(new_weights) == len(self.scale_weights), "Number of weights must match"
        self.scale_weights = new_weights
