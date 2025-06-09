import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from PIL import Image
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from DPT_blur.data_loader import BlurMapDataset


def load_val_subset(val_dir, crop_size, num_samples=50, start_idx=0):
    from torchvision import transforms

    class MIMONormalize:
        def __call__(self, sample):
            image_np = sample["image"].astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float()
            sample["image"] = image_tensor - 0.5
            return sample

    transform = transforms.Compose([MIMONormalize()])

    dataset = BlurMapDataset(
        blurred_dir=os.path.join(val_dir, 'blur'),
        gt_dir=os.path.join(val_dir, 'condition'),
        transform=transform,
        crop_size=crop_size,
        is_train=False
    )

    end_idx = min(start_idx + num_samples, len(dataset))
    indices = list(range(start_idx, end_idx))
    return Subset(dataset, indices)


def display_samples_grid(dataset, output_path=None, show_magnitude=False):
    n = len(dataset)
    cols = 10
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i in range(rows * cols):
        ax = axs[i // cols, i % cols] if rows > 1 else axs[i]
        ax.axis('off')

        if i >= n:
            continue

        sample = dataset[i]

        if isinstance(sample, dict):
            img = sample['blur']
            blur_field = sample['blur_field']
        else:
            img = sample[0]
            blur_field = sample[1]

        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np + 0.5, 0, 1)

        if show_magnitude:
            mag = blur_field[2].numpy()
            ax.imshow(mag, cmap='viridis')
        else:
            ax.imshow(img_np)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Grid saved to {output_path}")

    plt.close()


def main(args):
    for i in range(3):
        start_idx = i * 50
        subset = load_val_subset(args.val_dir, args.crop_size, num_samples=50, start_idx=start_idx)

        input_grid_path = os.path.join(args.output_dir, f"input_grid_{i}.png")
        mag_grid_path = os.path.join(args.output_dir, f"mag_grid_{i}.png")

        display_samples_grid(subset, output_path=input_grid_path, show_magnitude=False)
        display_samples_grid(subset, output_path=mag_grid_path, show_magnitude=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='grids', help='Directory to save image and magnitude grids')
    args = parser.parse_args()

    main(args)
