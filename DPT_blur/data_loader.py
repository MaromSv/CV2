import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import os
import glob
import random

class BlurMapDataset(Dataset):
    """
    Dataset for loading blurred images and their corresponding blur map ground truth.

    Loads image-ground_truth pairs and applies transformations.
    Includes optional random cropping for training augmentation.
    """
    def __init__(self, blurred_dir, gt_dir, transform=None, target_transform=None, crop_size=None, is_train=False):
        """
        Args:
            blurred_dir (str): Directory containing blurred input images (.png, .jpg, .jpeg).
            gt_dir (str): Directory containing ground truth blur maps (.npy files).
                          Expected GT format is (bx, by) with shape (2, H, W).
            transform (callable, optional): Optional transform to be applied to the input image sample dictionary.
                                           Expected input: {'image': image_array}
                                           Expected output: {'image': image_tensor}
            target_transform (callable, optional): Optional transform to be applied to the ground truth tensor.
            crop_size (int, optional): The desired height and width for random cropping during training.
                                       If None or is_train is False, no cropping is performed. Defaults to None.
            is_train (bool, optional): If True, enables random cropping (if crop_size is set) and potentially
                                       other training-specific augmentations within the transform pipeline.
                                       Defaults to False.
        """
        self.blurred_dir = blurred_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.target_transform = target_transform
        self.crop_size = crop_size
        self.is_train = is_train

        # Find all image files in the directory
        self.image_files = sorted(glob.glob(os.path.join(blurred_dir, '*.png'))) \
                         + sorted(glob.glob(os.path.join(blurred_dir, '*.jpg'))) \
                         + sorted(glob.glob(os.path.join(blurred_dir, '*.jpeg')))

        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {blurred_dir}")

        print(f"Found {len(self.image_files)} potential image files.")

        # TODO (Optional Pre-filtering):
        # Consider implementing pre-filtering to ensure only images with corresponding
        # GT files are included in self.image_files for efficiency.
        # self.image_files = self._pre_filter_pairs(self.image_files, self.gt_dir)
        # print(f"Found {len(self.image_files)} valid image/GT pairs.")

    # Optional pre-filtering method skeleton
    # def _pre_filter_pairs(self, image_files, gt_dir):
    #     valid_files = []
    #     print("Pre-filtering image/GT pairs...")
    #     for img_path in tqdm(image_files):
    #         base_name = os.path.splitext(os.path.basename(img_path))[0]
    #         gt_path = os.path.join(gt_dir, base_name + '.npy')
    #         if os.path.exists(gt_path):
    #             valid_files.append(img_path)
    #     return valid_files

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
                   gt_tensor: Ground truth blur map tensor, resized to match image_tensor shape.
            None: If any error occurs during loading, processing, or if filtering is needed.
        """
        img_path = self.image_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(self.gt_dir, base_name + '.npy')

        # Check if GT file exists (redundant if pre-filtering is implemented)
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
            # TODO: Adapt GT loading based on the final format.
            # This implementation assumes GT is ALREADY stored as (bx, by) in .npy files.
            # Expected shape: (2, H, W)
            gt_map_xy = np.load(gt_path).astype(np.float32) # CHW

            if gt_map_xy.ndim != 3 or gt_map_xy.shape[0] != 2:
                 raise ValueError(f"Expected GT shape (2, H, W) for (bx, by), got {gt_map_xy.shape} for {gt_path}")

            # TODO: If GT is stored differently (e.g., Magnitude/Angle), implement conversion here.
            # Example (if GT was [Mag, Angle_Radians]):
            # if gt_map_mag_angle.shape[0] != 2:
            #     raise ValueError(f"Expected GT shape (2, H, W), got {gt_map_mag_angle.shape}")
            # print(f"DEBUG: Converting GT {base_name}.npy from (Mag, Angle) -> (bx, by)")
            # magnitude = gt_map_mag_angle[0, :, :]
            # angle_rad = gt_map_mag_angle[1, :, :] # Ensure this is in RADIANS
            # bx = magnitude * np.cos(angle_rad)
            # by = magnitude * np.sin(angle_rad)
            # gt_map_xy = np.stack((bx, by), axis=0) # CHW

        except Exception as e:
             print(f"Error loading/processing ground truth {gt_path}: {e}. Returning None.")
             return None

        # --- Random Cropping (for training) ---
        if self.is_train and self.crop_size is not None:
            H_orig, W_orig = image.shape[:2] # Original image dimensions (HWC)
            C_gt, H_gt, W_gt = gt_map_xy.shape # Original GT dimensions (CHW)

            # Basic check for dimension consistency
            if H_orig != H_gt or W_orig != W_gt:
                print(f"Warning: Mismatch in original image ({H_orig}x{W_orig}) and GT ({H_gt}x{W_gt}) dimensions for {base_name}. Skipping.")
                return None

            # Check if image is large enough for cropping
            if H_orig < self.crop_size or W_orig < self.crop_size:
                print(f"Warning: Image {base_name} ({H_orig}x{W_orig}) is smaller than crop size ({self.crop_size}). Skipping.")
                return None

            # Calculate random top-left corner for the crop
            top = random.randint(0, H_orig - self.crop_size)
            left = random.randint(0, W_orig - self.crop_size)

            # Perform the crop on both image and GT map
            image = image[top : top + self.crop_size, left : left + self.crop_size, :] # HWC
            gt_map_xy = gt_map_xy[:, top : top + self.crop_size, left : left + self.crop_size] # CHW

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
        gt_tensor = torch.from_numpy(gt_map_xy.copy()).float() # Use .copy() for safety

        # Resize GT map tensor to match the final spatial dimensions of the image_tensor
        # This accounts for any resizing done within the self.transform pipeline (e.g., ensure_multiple_of)
        final_target_size = image_tensor.shape[1:] # Get (H, W) from image_tensor (C, H, W)
        if gt_tensor.shape[1:] != final_target_size:
            gt_tensor = TF.interpolate(gt_tensor.unsqueeze(0), size=final_target_size, mode='bilinear', align_corners=False)
            gt_tensor = gt_tensor.squeeze(0) # Remove batch dim added for interpolate

        # Apply optional target-specific transforms (e.g., normalization/scaling of bx, by)
        if self.target_transform:
             gt_tensor = self.target_transform(gt_tensor)

        return image_tensor, gt_tensor 