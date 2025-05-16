import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F_nn
import numpy as np
import cv2
import os
import glob
import random

class BlurMapDataset(Dataset):
    """Dataset for blur field prediction"""
    
    def __init__(self, root_dir, transform=None, crop_size=256):
        """
        Args:
            root_dir (str): Directory with blur images and blur field maps
                            Expected structure: root_dir/blur/ and root_dir/condition/
            transform (callable, optional): Optional transform to be applied on a sample
            crop_size (int): Size of random crops during training
        """
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        
        # Get blur image paths
        self.blur_dir = os.path.join(root_dir, 'blur')
        self.blur_field_dir = os.path.join(root_dir, 'condition')
        
        # Check if directories exist
        if not os.path.exists(self.blur_dir):
            raise ValueError(f"Blur directory not found: {self.blur_dir}")
        if not os.path.exists(self.blur_field_dir):
            raise ValueError(f"Blur field directory not found: {self.blur_field_dir}")
        
        # Get image filenames - support both png and jpg formats
        self.image_list = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_list.extend(glob.glob(os.path.join(self.blur_dir, ext)))
        self.image_list = sorted([os.path.basename(f) for f in self.image_list])
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # Load blur image
        img_name = self.image_list[idx]
        blur_img_path = os.path.join(self.blur_dir, img_name)
        blur_img = cv2.imread(blur_img_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Load blur field (bx, by, magnitude)
        blur_field_path = os.path.join(self.blur_field_dir, img_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        
        if os.path.exists(blur_field_path):
            blur_field = np.load(blur_field_path)
        else:
            # If .npy doesn't exist, try loading as image
            blur_field_img_path = os.path.join(self.blur_field_dir, img_name)
            if os.path.exists(blur_field_img_path):
                blur_field_img = cv2.imread(blur_field_img_path)
                blur_field_img = cv2.cvtColor(blur_field_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                # Assuming RGB channels represent bx, by, magnitude
                blur_field = np.transpose(blur_field_img, (2, 0, 1))
            else:
                raise ValueError(f"Blur field not found for {img_name}")
        
        # Make sure blur_field has shape [3, H, W]
        if blur_field.shape[0] != 3:
            if blur_field.shape[-1] == 3:  # If shape is [H, W, 3]
                blur_field = np.transpose(blur_field, (2, 0, 1))
        
        # Random crop during training if crop_size is specified
        if self.crop_size > 0:
            h, w = blur_img.shape[:2]
            
            # Ensure crop size is not larger than image
            crop_size = min(self.crop_size, h, w)
            
            # Random crop coordinates
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            
            # Apply crop
            blur_img = blur_img[top:top+crop_size, left:left+crop_size]
            blur_field = blur_field[:, top:top+crop_size, left:left+crop_size]
        
        # Convert to tensors
        blur_img = torch.from_numpy(blur_img.transpose(2, 0, 1)).float()
        blur_field = torch.from_numpy(blur_field).float()
        
        # Normalize images to [-0.5, 0.5] range
        blur_img = blur_img - 0.5
        
        # Apply additional transforms if provided
        if self.transform:
            blur_img = self.transform(blur_img)
        
        return {
            'blur': blur_img,
            'blur_field': blur_field
        }
