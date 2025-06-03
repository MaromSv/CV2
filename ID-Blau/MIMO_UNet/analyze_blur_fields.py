import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def analyze_blur_fields(data_dir, output_dir=None, num_samples=10, create_histograms=True):
    """
    Analyzes .npy files in the given directory to provide statistics about blur fields.
    
    Args:
        data_dir (str): Directory containing .npy files
        output_dir (str): Directory to save histograms and statistics
        num_samples (int): Number of samples to analyze (-1 for all)
        create_histograms (bool): Whether to create histograms
    """
    print(f"\nAnalyzing blur fields in: {data_dir}")
    
    # Find all .npy files in the directory
    npy_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
    
    if not npy_files:
        print(f"No .npy files found in {data_dir}")
        return
    
    # Limit the number of samples if specified
    if num_samples > 0 and num_samples < len(npy_files):
        print(f"Found {len(npy_files)} .npy files. Analyzing {num_samples} samples...")
        npy_files = npy_files[:num_samples]
    else:
        print(f"Found {len(npy_files)} .npy files. Analyzing all samples...")
    
    # Channel names for blur fields
    channel_names = ["bx (cos)", "by (sin)", "magnitude"]
    
    # Arrays to store statistics
    all_mins = []
    all_maxs = []
    all_means = []
    all_stds = []
    all_values = [[] for _ in range(3)]  # One list for each channel
    
    # Process each file
    for i, npy_file in enumerate(npy_files):
        try:
            # Load the .npy file
            data = np.load(npy_file)
            
            # Print basic information
            print(f"\nFile {i+1}/{len(npy_files)}: {os.path.basename(npy_file)}")
            print(f"  Shape: {data.shape}, Type: {data.dtype}")
            
            # Check if the data has the expected shape (3, H, W)
            if data.ndim != 3 or data.shape[0] != 3:
                print(f"  WARNING: Unexpected shape. Expected (3, H, W), got {data.shape}")
                continue
            
            # Calculate statistics for each channel
            file_stats = {"mins": [], "maxs": [], "means": [], "stds": []}
            
            for ch_idx in range(3):
                channel_data = data[ch_idx]
                
                # Calculate statistics
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                
                # Store statistics
                file_stats["mins"].append(min_val)
                file_stats["maxs"].append(max_val)
                file_stats["means"].append(mean_val)
                file_stats["stds"].append(std_val)
                
                # Sample values for histogram (to avoid memory issues)
                if create_histograms:
                    flat_data = channel_data.flatten()
                    if len(flat_data) > 10000:
                        indices = np.random.choice(len(flat_data), 10000, replace=False)
                        all_values[ch_idx].extend(flat_data[indices])
                    else:
                        all_values[ch_idx].extend(flat_data)
                
                # Print channel statistics
                print(f"  Channel {ch_idx} ({channel_names[ch_idx]}):")
                print(f"    Min: {min_val:.6f}, Max: {max_val:.6f}")
                print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            
            # Add this file's stats to the overall collection
            all_mins.append(file_stats["mins"])
            all_maxs.append(file_stats["maxs"])
            all_means.append(file_stats["means"])
            all_stds.append(file_stats["stds"])
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(npy_file)}: {e}")
    
    # Calculate overall statistics
    if all_mins:
        all_mins = np.array(all_mins)
        all_maxs = np.array(all_maxs)
        all_means = np.array(all_means)
        all_stds = np.array(all_stds)
        
        print("\n=== OVERALL STATISTICS ===")
        for ch_idx in range(3):
            print(f"\nChannel {ch_idx} ({channel_names[ch_idx]}):")
            print(f"  Min range: {np.min(all_mins[:, ch_idx]):.6f} to {np.max(all_mins[:, ch_idx]):.6f}")
            print(f"  Max range: {np.min(all_maxs[:, ch_idx]):.6f} to {np.max(all_maxs[:, ch_idx]):.6f}")
            print(f"  Overall min: {np.min(all_mins[:, ch_idx]):.6f}, Overall max: {np.max(all_maxs[:, ch_idx]):.6f}")
            print(f"  Mean of means: {np.mean(all_means[:, ch_idx]):.6f}")
            print(f"  Mean of standard deviations: {np.mean(all_stds[:, ch_idx]):.6f}")
    
    # Create histograms if requested
    if create_histograms and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual histograms for each channel
        for ch_idx in range(3):
            if all_values[ch_idx]:
                plt.figure(figsize=(10, 6))
                sns.histplot(all_values[ch_idx], bins=100, kde=True)
                plt.title(f"Distribution of {channel_names[ch_idx]} values")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f"histogram_channel_{ch_idx}_{channel_names[ch_idx].replace(' ', '_')}.png"), dpi=300)
                plt.close()
        
        # Create a combined plot with all channels
        plt.figure(figsize=(12, 8))
        for ch_idx in range(3):
            if all_values[ch_idx]:
                sns.kdeplot(all_values[ch_idx], label=channel_names[ch_idx])
        plt.title("Distribution of values across all channels")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "histogram_all_channels.png"), dpi=300)
        plt.close()
        
        # Save statistics to a text file
        with open(os.path.join(output_dir, "statistics.txt"), "w") as f:
            f.write(f"Statistics for {len(npy_files)} .npy files in {data_dir}\n\n")
            for ch_idx in range(3):
                f.write(f"Channel {ch_idx} ({channel_names[ch_idx]}):\n")
                f.write(f"  Min range: {np.min(all_mins[:, ch_idx]):.6f} to {np.max(all_mins[:, ch_idx]):.6f}\n")
                f.write(f"  Max range: {np.min(all_maxs[:, ch_idx]):.6f} to {np.max(all_maxs[:, ch_idx]):.6f}\n")
                f.write(f"  Overall min: {np.min(all_mins[:, ch_idx]):.6f}, Overall max: {np.max(all_maxs[:, ch_idx]):.6f}\n")
                f.write(f"  Mean of means: {np.mean(all_means[:, ch_idx]):.6f}\n")
                f.write(f"  Mean of standard deviations: {np.mean(all_stds[:, ch_idx]):.6f}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze blur field .npy files")
    parser.add_argument("--train_dir", type=str, help="Path to training dataset directory")
    parser.add_argument("--val_dir", type=str, help="Path to validation dataset directory")
    parser.add_argument("--output_dir", type=str, default="./blur_field_stats", help="Directory to save statistics and histograms")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to analyze (-1 for all)")
    parser.add_argument("--no_histograms", action="store_true", help="Skip creating histograms")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze training data if provided
    if args.train_dir:
        train_output_dir = os.path.join(args.output_dir, "train")
        analyze_blur_fields(args.train_dir, train_output_dir, args.num_samples, not args.no_histograms)
    
    # Analyze validation data if provided
    if args.val_dir:
        val_output_dir = os.path.join(args.output_dir, "val")
        analyze_blur_fields(args.val_dir, val_output_dir, args.num_samples, not args.no_histograms)
    
    # If neither is provided, print help
    if not args.train_dir and not args.val_dir:
        parser.print_help()