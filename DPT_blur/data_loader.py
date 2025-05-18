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
    """
    Dataset for loading blurred images and their corresponding blur map ground truth.

    Loads image-ground_truth pairs and applies transformations.
    Includes optional random cropping and random horizontal flip for training augmentation.
    """
    def __init__(self, blurred_dir, gt_dir, transform=None, target_transform=None, crop_size=None, is_train=False, random_flip=False):
        """
        Args:
            blurred_dir (str): Directory containing blurred input images (.png, .jpg, .jpeg).
            gt_dir (str): Directory containing ground truth blur maps (.npy files).
                          Expected GT format is (bx, by, magnitude) with shape (3, H, W).
            transform (callable, optional): Optional transform to be applied to the input image sample dictionary.
                                           Expected input: {'image': image_array}
                                           Expected output: {'image': image_tensor}
            target_transform (callable, optional): Optional transform to be applied to the ground truth tensor.
            crop_size (int, optional): The desired height and width for random cropping during training.
                                       If None or is_train is False, no cropping is performed. Defaults to None.
            is_train (bool, optional): If True, enables random cropping (if crop_size is set) and potentially
                                       other training-specific augmentations within the transform pipeline.
                                       Defaults to False.
            random_flip (bool, optional): If True and is_train is True, enables random horizontal flipping.
        """
        self.blurred_dir = blurred_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.target_transform = target_transform
        self.crop_size = crop_size
        self.is_train = is_train
        self.random_flip = random_flip

        # Find all image files directly in the blurred_dir
        self.image_files = []
        extensions = ['png', 'jpg', 'jpeg']
        for ext in extensions:
            pattern = os.path.join(self.blurred_dir, f'*.{ext}')
            self.image_files.extend(sorted(glob.glob(pattern, recursive=False))) # recursive=False for flat dir

        # Ensure the list is sorted for reproducibility if multiple extensions are mixed
        self.image_files.sort()

        if not self.image_files:
            # More specific error message for the new structure
            raise FileNotFoundError(f"No image files (png, jpg, jpeg) found directly in {self.blurred_dir}")

        print(f"Found {len(self.image_files)} image files in {self.blurred_dir}.")

    def __len__(self):
        """Returns the total number of image files found."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads and processes a single sample (image and ground truth).

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, gt_tensor) if successful, otherwise None.
                   image_tensor: Transformed input image (potentially cropped).
                   gt_tensor: Ground truth blur map tensor (bx, by, magnitude),
                              resized to match image_tensor shape.
            None: If any error occurs during loading, processing, or if filtering is needed.
        """
        img_path = self.image_files[idx]

        # Construct GT path based on the image file name
        try:
            image_basename = os.path.basename(img_path) # e.g., "image_000001.png"
            gt_filename_base = os.path.splitext(image_basename)[0] # e.g., "image_000001"
            gt_path = os.path.join(self.gt_dir, gt_filename_base + '.npy') # e.g., gt_dir/image_000001.npy
        except Exception as e:
            print(f"Error constructing GT path for {img_path} from gt_dir {self.gt_dir}: {e}. Returning None.")
            return None

        # Check if GT file exists
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for {img_path} (Index {idx}). Returning None.")
            return None

        # --- Load Image ---
        try:
            image = cv2.imread(img_path) # HWC, BGR
            if image is None: raise IOError(f"imread failed for {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # HWC, RGB
            image = image.astype(np.float32)
        except Exception as e:
             print(f"Error loading image {img_path}: {e}. Returning None.")
             return None

        # --- Load Ground Truth ---
        try:
            # Load the 3-channel (bx, by, magnitude) .npy file
            # Expected shape: (3, H, W)
            gt_blur_map = np.load(gt_path).astype(np.float32) # CHW

            if gt_blur_map.ndim != 3 or gt_blur_map.shape[0] != 3:
                 raise ValueError(f"Expected GT shape (3, H, W) for (bx, by, magnitude), got {gt_blur_map.shape} for {gt_path}")

        except Exception as e:
             print(f"Error loading/processing ground truth {gt_path}: {e}. Returning None.")
             return None

        # --- Cropping (applied if self.crop_size is set) ---
        if self.crop_size is not None:
            H_orig, W_orig = image.shape[:2] # Original image dimensions (HWC)
            C_gt, H_gt, W_gt = gt_blur_map.shape # Original GT dimensions (CHW)

            if H_orig != H_gt or W_orig != W_gt:
                print(f"Warning: Mismatch in original image ({H_orig}x{W_orig}) and GT ({H_gt}x{W_gt}) dimensions for {os.path.basename(img_path)}. Skipping.")
                return None

            if H_orig < self.crop_size or W_orig < self.crop_size:
                print(f"Warning: Image {os.path.basename(img_path)} ({H_orig}x{W_orig}) is smaller than crop size ({self.crop_size}). Skipping.")
                return None

            if self.is_train: # Random crop for training
                top = random.randint(0, H_orig - self.crop_size)
                left = random.randint(0, W_orig - self.crop_size)
            else: # Center crop for validation (if crop_size is provided and not is_train)
                top = (H_orig - self.crop_size) // 2
                left = (W_orig - self.crop_size) // 2
            
            image = image[top : top + self.crop_size, left : left + self.crop_size, :] # HWC
            gt_blur_map = gt_blur_map[:, top : top + self.crop_size, left : left + self.crop_size] # CHW

        # --- Random Horizontal Flip (for training, applied to both image and GT) ---
        if self.is_train and self.random_flip and random.random() < 0.5:
            image = cv2.flip(image, 1) # HWC
            gt_blur_map = np.ascontiguousarray(gt_blur_map[:, :, ::-1]) # CHW, flip horizontally (axis 2)
            # For bx (channel 0 of gt_blur_map), its sign needs to be inverted after a horizontal flip
            gt_blur_map[0, :, :] *= -1

        # --- Apply Input Image Transforms ---
        # The transform pipeline (e.g., dpt_transform) should handle:
        # 1. Any further augmentations (flips, rotates) - ensure they are applied consistently if needed for GT.
        # 2. Normalization (e.g., NormalizeImage)
        # 3. Conversion to CHW tensor format (e.g., PrepareForNet)
        if self.transform:
            # The transform should expect a dictionary {'image': HWC_image_array}
            # and return a dictionary {'image': CHW_image_tensor}
            sample = {"image": image}
            transformed_sample = self.transform(sample)
            image_tensor = transformed_sample["image"] # Should be CHW tensor
        else:
            # Basic fallback: Convert to tensor and maybe normalize manually
            image_tensor = TF.to_tensor(image) # Converts HWC uint8/float32 -> CHW float[0,1]
            # Example normalization: image_tensor = TF.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # --- Process Ground Truth Tensor ---
        # Convert GT numpy array (CHW) to tensor
        gt_tensor = torch.from_numpy(gt_blur_map.copy()).float() # Use .copy() for safety

        # Resize GT map tensor to match the final spatial dimensions of the image_tensor
        # This accounts for any resizing done within the self.transform pipeline (e.g., ensure_multiple_of)
        final_target_size = image_tensor.shape[1:] # Get (H, W) from image_tensor (C, H, W)
        if gt_tensor.shape[1:] != final_target_size:
            gt_tensor = F_nn.interpolate(gt_tensor.unsqueeze(0), size=final_target_size, mode='bilinear', align_corners=False)
            gt_tensor = gt_tensor.squeeze(0) # Remove batch dim added for interpolate

        # Apply optional target-specific transforms (e.g., normalization/scaling of bx, by, magnitude)
        if self.target_transform:
             gt_tensor = self.target_transform(gt_tensor)

        return image_tensor, gt_tensor 
