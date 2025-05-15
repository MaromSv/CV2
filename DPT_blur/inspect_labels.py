import numpy as np
import os
import glob

def inspect_dataset_labels(condition_dir, num_samples=5):
    """
    Loads a few .npy label files from the specified directory and prints their shapes,
    mean, variance, min, max, and NaN counts for each channel.

    Args:
        condition_dir (str): Path to the directory containing .npy label files.
        num_samples (int): Number of samples to inspect.
    """
    print(f"Inspecting labels in: {condition_dir}")

    if not os.path.isdir(condition_dir):
        print(f"Error: Directory not found: {condition_dir}")
        return

    npy_files = sorted(glob.glob(os.path.join(condition_dir, "*.npy")))

    if not npy_files:
        print(f"No .npy files found in {condition_dir}")
        return

    print(f"Found {len(npy_files)} .npy files. Inspecting up to {num_samples} samples...")

    channel_names = ["cos_component", "sin_component", "magnitude"] # Updated channel names

    for i, npy_file_path in enumerate(npy_files[:num_samples]):
        try:
            label_data = np.load(npy_file_path) # Expected shape (C, H, W)
            print(f"\n  --- File: {os.path.basename(npy_file_path)} ---")
            print(f"    Shape: {label_data.shape}, Data type: {label_data.dtype}")

            if label_data.ndim == 3:
                num_channels = label_data.shape[0]
                for ch_idx in range(num_channels):
                    channel_data = label_data[ch_idx, :, :]
                    ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Channel {ch_idx}"
                    
                    # Use np.nanmean, np.nanvar, np.nanmin, np.nanmax to ignore NaNs in stats
                    mean_val = np.nanmean(channel_data)
                    var_val = np.nanvar(channel_data)
                    min_val = np.nanmin(channel_data)
                    max_val = np.nanmax(channel_data)
                    nan_count = np.sum(np.isnan(channel_data))
                    total_elements = channel_data.size
                    nan_percentage = (nan_count / total_elements) * 100 if total_elements > 0 else 0
                    
                    print(f"    Channel {ch_idx} ({ch_name}):")
                    print(f"      Mean (ignoring NaNs): {mean_val:.4f}, Variance (ignoring NaNs): {var_val:.4f}")
                    print(f"      Min (ignoring NaNs):  {min_val:.4f}, Max (ignoring NaNs):      {max_val:.4f}")
                    print(f"      NaN count: {nan_count} / {total_elements} ({nan_percentage:.2f}%)")
            else:
                print(f"    Label data is not 3-dimensional (C, H, W). Skipping channel-wise stats.")

        except Exception as e:
            print(f"  - Error processing {os.path.basename(npy_file_path)}: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Assuming this script is in CV2_code/CV2/DPT_blur/
    # and the dataset is at CV2_code/CV2/DPT_blur/data/dataset_DPT_blur/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the training condition maps in the restructured dataset
    # Modify if your script or dataset location is different
    DEFAULT_TRAIN_CONDITION_DIR = os.path.join(current_script_dir, "data", "dataset_DPT_blur", "train", "condition")

    # --- Run Inspection ---
    inspect_dataset_labels(DEFAULT_TRAIN_CONDITION_DIR, num_samples=10) # Increased num_samples for better check

    # You can also inspect validation labels if needed:
    # DEFAULT_VAL_CONDITION_DIR = os.path.join(current_script_dir, "data", "dataset_DPT_blur", "val", "condition")
    # print("\nInspecting validation labels:")
    # inspect_dataset_labels(DEFAULT_VAL_CONDITION_DIR, num_samples=10) 