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
