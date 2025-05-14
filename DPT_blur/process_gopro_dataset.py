import os
import glob
import shutil
import argparse
import numpy as np
import cv2
from tqdm import tqdm

def process_gopro_dataset(gopro_root_dir, output_root_dir):
    """
    Processes the GOPRO dataset into a new structure suitable for training.

    The GOPRO dataset is expected to have a structure like:
    gopro_root_dir/
        train/
            SCENE_NAME_1/
                blur/
                    image1.png
                    ...
                sharp/
                    ...
            SCENE_NAME_2/
                ...
        test/
            SCENE_NAME_3/
                blur/
                    imageN.png
                    ...
                sharp/
                    ...

    The output structure will be:
    output_root_dir/
        train_images/
            SCENE_NAME_1/
                blur/
                    image1.png
                    ...
        train_gt/
            SCENE_NAME_1/
                blur/
                    image1.npy (dummy GT)
                    ...
        val_images/ (from GOPRO test set)
            SCENE_NAME_3/
                blur/
                    imageN.png
                    ...
        val_gt/ (from GOPRO test set)
            SCENE_NAME_3/
                blur/
                    imageN.npy (dummy GT)
                    ...
    """
    print(f"Processing GOPRO dataset from: {gopro_root_dir}")
    print(f"Output will be saved to: {output_root_dir}")

    set_types_map = {
        "train": ("train_images", "train_gt"),
        "test": ("val_images", "val_gt") # GOPRO 'test' maps to 'val' for our training
    }

    for gopro_set_type, (out_img_folder_name, out_gt_folder_name) in set_types_map.items():
        current_gopro_set_path = os.path.join(gopro_root_dir, gopro_set_type)
        if not os.path.isdir(current_gopro_set_path):
            print(f"Warning: GOPRO set directory not found: {current_gopro_set_path}. Skipping.")
            continue

        output_images_base_path = os.path.join(output_root_dir, out_img_folder_name)
        output_gt_base_path = os.path.join(output_root_dir, out_gt_folder_name)

        os.makedirs(output_images_base_path, exist_ok=True)
        os.makedirs(output_gt_base_path, exist_ok=True)

        print(f"Processing {gopro_set_type} set (outputting to {out_img_folder_name} and {out_gt_folder_name})...")

        scene_folders = sorted([d for d in os.listdir(current_gopro_set_path) if os.path.isdir(os.path.join(current_gopro_set_path, d))])

        if not scene_folders:
            print(f"No scene folders found in {current_gopro_set_path}.")
            continue
            
        for scene_name in tqdm(scene_folders, desc=f"Scenes in {gopro_set_type}"):
            gopro_scene_blur_path = os.path.join(current_gopro_set_path, scene_name, "blur")

            if not os.path.isdir(gopro_scene_blur_path):
                # print(f"No 'blur' subfolder in scene {scene_name} of {gopro_set_type} set. Path: {gopro_scene_blur_path}. Skipping scene.")
                continue

            # Output paths for this scene
            output_scene_images_path = os.path.join(output_images_base_path, scene_name, "blur")
            output_scene_gt_path = os.path.join(output_gt_base_path, scene_name, "blur")

            os.makedirs(output_scene_images_path, exist_ok=True)
            os.makedirs(output_scene_gt_path, exist_ok=True)

            image_files = []
            image_files.extend(sorted(glob.glob(os.path.join(gopro_scene_blur_path, '*.png'))))
            image_files.extend(sorted(glob.glob(os.path.join(gopro_scene_blur_path, '*.jpg'))))
            image_files.extend(sorted(glob.glob(os.path.join(gopro_scene_blur_path, '*.jpeg'))))
            
            if not image_files:
                # print(f"No images found in {gopro_scene_blur_path}. Skipping.")
                continue

            for img_src_path in image_files:
                img_basename = os.path.basename(img_src_path)
                img_name_no_ext = os.path.splitext(img_basename)[0]

                # Copy image
                img_dest_path = os.path.join(output_scene_images_path, img_basename)
                try:
                    shutil.copy2(img_src_path, img_dest_path)
                except Exception as e:
                    print(f"Error copying image {img_src_path} to {img_dest_path}: {e}")
                    continue

                # Create and save dummy GT .npy file
                try:
                    image = cv2.imread(img_src_path) # Read to get dimensions
                    if image is None:
                        print(f"Warning: Could not read image {img_src_path} to determine dimensions for GT. Skipping GT generation.")
                        continue
                    
                    H, W, _ = image.shape
                    dummy_gt = np.zeros((3, H, W), dtype=np.float32) # Shape (3, H, W)

                    gt_dest_path = os.path.join(output_scene_gt_path, img_name_no_ext + '.npy')
                    np.save(gt_dest_path, dummy_gt)

                except Exception as e:
                    print(f"Error creating dummy GT for {img_src_path}: {e}")

    print("\\nDataset processing finished.")
    print(f"Processed data saved in {output_root_dir}")
    print("You can now use the following paths for your training script:")
    print(f"  --blurred_dir_train {os.path.join(output_root_dir, 'train_images')}")
    print(f"  --gt_dir_train      {os.path.join(output_root_dir, 'train_gt')}")
    print(f"  --blurred_dir_val   {os.path.join(output_root_dir, 'val_images')}")
    print(f"  --gt_dir_val        {os.path.join(output_root_dir, 'val_gt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GOPRO dataset into a structured format for DPT blur training.")
    parser.add_argument('--gopro_root', type=str, required=True, 
                        help="Root directory of the GOPRO dataset (e.g., 'DPT_blur/data/GOPRO_Large').")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory where the processed dataset will be saved (e.g., 'DPT_blur/data_processed').")
    
    args = parser.parse_args()

    if not os.path.isdir(args.gopro_root):
        print(f"Error: GOPRO root directory not found: {args.gopro_root}")
        exit(1)
        
    process_gopro_dataset(args.gopro_root, args.output_dir)

# Example usage from your workspace root:
# python DPT_blur/process_gopro_dataset.py --gopro_root DPT_blur/data/GOPRO_Large --output_dir DPT_blur/data_processed 