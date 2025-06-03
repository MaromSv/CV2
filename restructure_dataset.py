import os
import glob
import shutil
import random
from tqdm import tqdm
import argparse

def restructure_idblau_for_dpt(
    source_dir_base,
    output_dir_base,
    train_split_ratio=0.8,
    random_seed=42
):
    """
    Restructures the ID-Blau generated dataset (GOPRO_Large_Reblur) into
    a train/validation split suitable for the DPT_blur model.

    Assumes GOPRO_Large_Reblur contains:
        blur/<original_idx>/<variant_idx>.png
        condition/<original_idx>/<variant_idx>.npy

    Creates a new structure:
        <output_dir_base>/
            train/
                blur/       (e.g., image_000001.png)
                condition/  (e.g., image_000001.npy)
            val/
                blur/
                condition/

    Args:
        source_dir_base (str): Path to the GOPRO_Large_Reblur directory.
        output_dir_base (str): Path to the directory where the restructured
                               dataset will be created (e.g., "dataset_DPT_blur").
        train_split_ratio (float): Proportion of data to use for training (0.0 to 1.0).
        random_seed (int): Seed for shuffling to ensure reproducibility.
    """
    # Convert relative paths to absolute paths at the beginning
    source_dir_base = os.path.abspath(source_dir_base)
    output_dir_base = os.path.abspath(output_dir_base)

    print(f"Source ID-Blau dataset: {source_dir_base}")
    print(f"Output restructured dataset: {output_dir_base}")

    source_blur_dir = os.path.join(source_dir_base, "blur")
    source_condition_dir = os.path.join(source_dir_base, "condition")

    if not os.path.isdir(source_blur_dir):
        print(f"Error: Source blur directory not found: {source_blur_dir}")
        return
    if not os.path.isdir(source_condition_dir):
        print(f"Error: Source condition directory not found: {source_condition_dir}")
        return

    # --- 1. Collect all corresponding blur/condition file pairs ---
    all_file_pairs = []
    print("Collecting file pairs...")

    original_idx_blur_dirs = sorted(glob.glob(os.path.join(source_blur_dir, "*")))

    for orig_blur_dir_path in tqdm(original_idx_blur_dirs, desc="Scanning source directories"):
        if not os.path.isdir(orig_blur_dir_path):
            continue
        original_idx = os.path.basename(orig_blur_dir_path)

        variant_blur_files = sorted(glob.glob(os.path.join(orig_blur_dir_path, "*.png")))

        for blur_file_path in variant_blur_files:
            variant_idx_with_ext = os.path.basename(blur_file_path) # e.g., 00000.png
            variant_idx = os.path.splitext(variant_idx_with_ext)[0] # e.g., 00000

            # Construct corresponding condition file path
            condition_file_path = os.path.join(source_condition_dir, original_idx, variant_idx + ".npy")

            if os.path.exists(condition_file_path):
                all_file_pairs.append({
                    "blur_original_path": blur_file_path,
                    "condition_original_path": condition_file_path,
                    "original_idx": original_idx,
                    "variant_idx": variant_idx
                })
            else:
                print(f"Warning: Condition file not found for {blur_file_path}. Expected at {condition_file_path}. Skipping pair.")

    if not all_file_pairs:
        print("Error: No valid blur/condition file pairs found. Please check the source dataset structure.")
        return

    print(f"Found {len(all_file_pairs)} valid blur/condition pairs.")

    # --- 2. Shuffle the pairs ---
    random.seed(random_seed)
    random.shuffle(all_file_pairs)
    print(f"Shuffled pairs using seed {random_seed}.")

    # --- 3. Split into train and validation sets ---
    num_train = int(len(all_file_pairs) * train_split_ratio)
    train_pairs = all_file_pairs[:num_train]
    val_pairs = all_file_pairs[num_train:]

    print(f"Splitting into {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs.")

    # --- 4. Create new directory structure and copy files ---
    os.makedirs(output_dir_base, exist_ok=True)

    set_counter = 0 # Global counter for unique filenames

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        print(f"\nProcessing {split_name} set...")

        output_split_dir = os.path.join(output_dir_base, split_name)
        output_blur_target_dir = os.path.join(output_split_dir, "blur")
        output_condition_target_dir = os.path.join(output_split_dir, "condition")

        os.makedirs(output_blur_target_dir, exist_ok=True)
        os.makedirs(output_condition_target_dir, exist_ok=True)

        for pair_info in tqdm(split_pairs, desc=f"Copying {split_name} files"):
            new_filename_base = f"image_{set_counter:07d}" # e.g., image_0000000

            # Copy blur image
            target_blur_path = os.path.join(output_blur_target_dir, new_filename_base + ".png")
            shutil.copy2(pair_info["blur_original_path"], target_blur_path)

            # Copy condition map
            target_condition_path = os.path.join(output_condition_target_dir, new_filename_base + ".npy")
            shutil.copy2(pair_info["condition_original_path"], target_condition_path)

            set_counter += 1

    print("\n--- Dataset Restructuring Complete ---")
    print(f"Total images processed: {set_counter}")
    print(f"Training images: {len(train_pairs)}")
    print(f"  - Blurred images: {os.path.join(output_dir_base, 'train', 'blur')}")
    print(f"  - Condition maps: {os.path.join(output_dir_base, 'train', 'condition')}")
    print(f"Validation images: {len(val_pairs)}")
    print(f"  - Blurred images: {os.path.join(output_dir_base, 'val', 'blur')}")
    print(f"  - Condition maps: {os.path.join(output_dir_base, 'val', 'condition')}")
    print("--------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restructures the ID-Blau generated dataset (GOPRO_Large_Reblur) "
                    "into a train/validation split for DPT_blur model."
    )
    parser.add_argument(
        "source_dir",
        type=str,
        help="Path to the source GOPRO_Large_Reblur directory "
             "(e.g., ../dataset/GOPRO_Large_Reblur)."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory where the restructured dataset will be created "
             "(e.g., ../dataset_DPT_blur)."
    )
    parser.add_argument(
        "--train_split_ratio",
        type=float,
        default=0.8,
        help="Proportion of data to use for training (default: 0.8)."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for shuffling to ensure reproducible splits (default: 42)."
    )

    args = parser.parse_args()

    # --- Run Restructuring ---
    restructure_idblau_for_dpt(
        source_dir_base=args.source_dir,
        output_dir_base=args.output_dir,
        train_split_ratio=args.train_split_ratio,
        random_seed=args.random_seed
    ) 