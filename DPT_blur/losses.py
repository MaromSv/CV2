import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurVectorLoss(nn.Module):
    """
    Composite loss function for blur vector field learning.
    Handles directional and magnitude components separately.
    
    The loss combines:
    1. Direction loss: Cosine similarity between predicted and target blur directions
    2. Magnitude loss: MSE between predicted and target blur magnitudes
    
    Args:
        lambda_dir (float): Weight for the directional component
        lambda_mag (float): Weight for the magnitude component
        epsilon (float): Small value to avoid division by zero
        use_magnitude_weighting (bool): If True, direction loss is weighted by magnitude
        magnitude_only (bool): If True, only use magnitude loss (ignore direction)
    """
    def __init__(self, lambda_dir=1.0, lambda_mag=1.0, epsilon=1e-6, 
                 use_magnitude_weighting=True, magnitude_only=False):
        super(BlurVectorLoss, self).__init__()
        self.lambda_dir = lambda_dir
        self.lambda_mag = lambda_mag
        self.epsilon = epsilon
        self.use_magnitude_weighting = use_magnitude_weighting
        self.magnitude_only = magnitude_only
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted blur vectors, shape [B, 3, H, W]
                                 Channel 0: bx (cosine component)
                                 Channel 1: by (sine component) 
                                 Channel 2: magnitude
            target (torch.Tensor): Target blur vectors, shape [B, 3, H, W]
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Extract components
        pred_bx, pred_by, pred_mag = pred[:, 0], pred[:, 1], pred[:, 2]
        target_bx, target_by, target_mag = target[:, 0], target[:, 1], target[:, 2]
        
        # Default values for direction loss
        direction_loss = torch.tensor(0.0, device=pred.device)
        
        # Only compute direction loss if not magnitude_only
        if not self.magnitude_only:
            # Normalize vectors for cosine similarity
            pred_norm = torch.sqrt(pred_bx**2 + pred_by**2 + self.epsilon)
            target_norm = torch.sqrt(target_bx**2 + target_by**2 + self.epsilon)
            
            pred_bx_norm = pred_bx / pred_norm
            pred_by_norm = pred_by / pred_norm
            target_bx_norm = target_bx / target_norm
            target_by_norm = target_by / target_norm
            
            # Compute cosine similarity
            cos_sim = pred_bx_norm * target_bx_norm + pred_by_norm * target_by_norm
            direction_loss = 1.0 - cos_sim.mean()
            
            # Optionally weight direction loss by magnitude
            if self.use_magnitude_weighting:
                direction_mask = (target_mag > self.epsilon)
                if direction_mask.sum() > 0:
                    direction_loss = direction_loss * target_mag
                    direction_loss = direction_loss.sum() / (target_mag.sum() + self.epsilon)
                else:
                    direction_loss = direction_loss.mean()
            
        # Magnitude loss (MSE)
        magnitude_loss = F.mse_loss(pred_mag, target_mag)
        
        # Combined loss
        lambda_dir_effective = 0.0 if self.magnitude_only else self.lambda_dir
        total_loss = lambda_dir_effective * direction_loss + self.lambda_mag * magnitude_loss
        
        # For debugging/monitoring
        loss_components = {
            'direction_loss': direction_loss.item() if not self.magnitude_only else 0.0,
            'magnitude_loss': magnitude_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components

def create_blur_vector_loss(lambda_dir=1.0, lambda_mag=1.0, 
                           use_magnitude_weighting=True, magnitude_only=False):
    """Helper function to create the blur vector loss instance"""
    return BlurVectorLoss(lambda_dir, lambda_mag, 
                         use_magnitude_weighting=use_magnitude_weighting,
                         magnitude_only=magnitude_only)

# Alternative simpler implementation for testing
def simple_blur_vector_loss(pred, target, lambda_dir=1.0, lambda_mag=1.0, magnitude_only=False):
    """
    A simpler implementation of the blur vector loss.
    More straightforward but less configurable than BlurVectorLoss.
    
    Args:
        pred, target: Tensors of shape [B, 3, H, W]
        lambda_dir, lambda_mag: Weights for direction and magnitude components
        magnitude_only: If True, only use magnitude loss
    
    Returns:
        Scalar loss value
    """
    # Extract components
    pred_bx, pred_by, pred_mag = pred[:, 0], pred[:, 1], pred[:, 2]
    target_bx, target_by, target_mag = target[:, 0], target[:, 1], target[:, 2]
    
    # Direction loss
    direction_loss = 0.0
    if not magnitude_only:
        # Direction loss using normalized vectors
        pred_norm = torch.sqrt(pred_bx**2 + pred_by**2 + 1e-6)
        target_norm = torch.sqrt(target_bx**2 + target_by**2 + 1e-6)
        
        pred_bx_norm = pred_bx / pred_norm
        pred_by_norm = pred_by / pred_norm
        target_bx_norm = target_bx / target_norm
        target_by_norm = target_by / target_norm
        
        direction_loss = 1.0 - (pred_bx_norm * target_bx_norm + pred_by_norm * target_by_norm).mean()
    
    # Magnitude loss
    magnitude_loss = F.mse_loss(pred_mag, target_mag)
    
    # Combined loss
    lambda_dir_effective = 0.0 if magnitude_only else lambda_dir
    return lambda_dir_effective * direction_loss + lambda_mag * magnitude_loss 

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class WeightedBlurVectorLoss(nn.Module):
    """
    Enhanced blur vector loss that combines:
    1. Weighted Charbonnier loss for overall vector field accuracy
    2. Weighted direction loss using cosine similarity
    3. Weighted magnitude loss for blur strength accuracy
    
    Args:
        lambda_dir (float): Weight for the directional component
        lambda_mag (float): Weight for the magnitude component
        epsilon (float): Small value to avoid division by zero
        use_magnitude_weighting (bool): If True, direction loss is weighted by magnitude
        magnitude_only (bool): If True, only use magnitude loss (ignore direction)
    """
    def __init__(self, lambda_dir=1.0, lambda_mag=1.0, epsilon=1e-6, 
                 use_magnitude_weighting=True, magnitude_only=False):
        super(WeightedBlurVectorLoss, self).__init__()
        self.lambda_dir = lambda_dir
        self.lambda_mag = lambda_mag
        self.epsilon = epsilon
        self.use_magnitude_weighting = use_magnitude_weighting
        self.magnitude_only = magnitude_only
        self.charbonnier = CharbonnierLoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted blur vectors, shape [B, 3, H, W]
                                 Channel 0: bx (cosine component)
                                 Channel 1: by (sine component) 
                                 Channel 2: magnitude
            target (torch.Tensor): Target blur vectors, shape [B, 3, H, W]
        
        Returns:
            tuple: (total_loss, loss_components_dict)
        """
        # Extract components
        pred_bx, pred_by, pred_mag = pred[:, 0], pred[:, 1], pred[:, 2]
        target_bx, target_by, target_mag = target[:, 0], target[:, 1], target[:, 2]
        
        # Charbonnier loss on the entire field
        charbonnier_loss = self.charbonnier(pred, target)
        
        # Default values for direction loss
        direction_loss = torch.tensor(0.0, device=pred.device)
        
        # Only compute direction loss if not magnitude_only
        if not self.magnitude_only:
            # Normalize vectors for cosine similarity
            pred_norm = torch.sqrt(pred_bx**2 + pred_by**2 + self.epsilon)
            target_norm = torch.sqrt(target_bx**2 + target_by**2 + self.epsilon)
            
            pred_bx_norm = pred_bx / pred_norm
            pred_by_norm = pred_by / pred_norm
            target_bx_norm = target_bx / target_norm
            target_by_norm = target_by / target_norm
            
            # Compute cosine similarity
            cos_sim = pred_bx_norm * target_bx_norm + pred_by_norm * target_by_norm
            direction_loss = 1.0 - cos_sim.mean()
            
            # Optionally weight direction loss by magnitude
            if self.use_magnitude_weighting:
                direction_mask = (target_mag > self.epsilon)
                if direction_mask.sum() > 0:
                    direction_loss = direction_loss * target_mag
                    direction_loss = direction_loss.sum() / (target_mag.sum() + self.epsilon)
                else:
                    direction_loss = direction_loss.mean()
        
        # Magnitude loss using Charbonnier
        magnitude_loss = self.charbonnier(pred_mag.unsqueeze(1), target_mag.unsqueeze(1))
        
        # Combined loss
        lambda_dir_effective = 0.0 if self.magnitude_only else self.lambda_dir
        total_loss = charbonnier_loss + lambda_dir_effective * direction_loss + self.lambda_mag * magnitude_loss
        
        # For debugging/monitoring
        loss_components = {
            'charbonnier_loss': charbonnier_loss.item(),
            'direction_loss': direction_loss.item() if not self.magnitude_only else 0.0,
            'magnitude_loss': magnitude_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components

def create_weighted_blur_vector_loss(lambda_dir=1.0, lambda_mag=1.0, 
                                    use_magnitude_weighting=True, magnitude_only=False):
    """Helper function to create the weighted blur vector loss instance"""
    return WeightedBlurVectorLoss(lambda_dir, lambda_mag, 
                                 use_magnitude_weighting=use_magnitude_weighting,
                                 magnitude_only=magnitude_only) 