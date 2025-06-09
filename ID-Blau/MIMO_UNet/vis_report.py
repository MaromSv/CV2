import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Add import path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))  # Assumes this script is inside ID-Blau

from DPT_blur.data_loader import BlurMapDataset
from DPT_blur.visualize_blur_map import visualize_multiple_blur_fields
from MIMOUNet import build_MIMOUnet_net
from random_model import RandomModel
from zero_model import ZeroModel


def load_validation_subset(val_dir, crop_size):
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

    indices = [206, 404, 148, 23, 146]  # Fixed validation indices
    return Subset(dataset, indices)


def visualize_direction_samples(model, dataset, device, output_dir):
    model.eval()
    pred_tensors, gt_tensors, input_images = [], [], []

    os.makedirs(output_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for sample in dataloader:
            if isinstance(sample, dict):
                blur_img = sample['blur'].to(device)
                blur_field = sample['blur_field'].to(device)
            else:
                blur_img = sample[0].to(device)
                blur_field = sample[1].to(device)

            out = model(blur_img)
            if isinstance(out, list):
                out = out[-1]
            if isinstance(out, tuple):
                out = torch.cat([out[0], out[1], out[2]], dim=1)

            pred_tensors.append(out[0].cpu())
            gt_tensors.append(blur_field[0].cpu())

            img_np = blur_img[0].permute(1, 2, 0).cpu().numpy()
            if img_np.min() < 0:
                img_np += 0.5
            img_np = np.clip(img_np, 0, 1)
            input_images.append(img_np)

    input_titles = [f"Input {i}" for i in range(len(input_images))]
    input_grid_path = os.path.join(output_dir, "input_grid.png")
    display_magnitude_grid(input_images, input_titles, input_grid_path, show_legend=False)

    visualize_multiple_blur_fields(pred_tensors, image_path_list=None, output_path=os.path.join(output_dir, "predictions_grid.png"))
    visualize_multiple_blur_fields(gt_tensors, image_path_list=None, output_path=os.path.join(output_dir, "gt_grid.png"))
    print(f"Saved direction prediction, GT, and input grids to {output_dir}")


def display_magnitude_grid(images_or_mags, titles, output_path, show_legend=False):
    import matplotlib.gridspec as gridspec

    cols = len(images_or_mags)
    fig = plt.figure(figsize=(cols * 3.5 + (1 if show_legend else 0), 4))
    gs = gridspec.GridSpec(1, cols + (1 if show_legend else 0), width_ratios=[1]*cols + ([0.05] if show_legend else []))

    ims = []
    for i, (img, title) in enumerate(zip(images_or_mags, titles)):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(img, cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis('off')
        ims.append(im)

    if show_legend:
        cbar_ax = fig.add_subplot(gs[-1])
        cbar = fig.colorbar(ims[-1], cax=cbar_ax)
        cbar.set_label('Magnitude')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved magnitude grid to {output_path}")


def visualize_magnitude_samples(model, dataset, device, output_dir):
    model.eval()
    input_images, gt_mags, pred_mags = [], [], []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for sample in dataloader:
            if isinstance(sample, dict):
                blur_img = sample['blur'].to(device)
                blur_field = sample['blur_field'].to(device)
            else:
                blur_img = sample[0].to(device)
                blur_field = sample[1].to(device)

            out = model(blur_img)
            if isinstance(out, list):
                out = out[-1]
            if isinstance(out, tuple):
                out = torch.cat([out[0], out[1], out[2]], dim=1)

            img_np = blur_img[0].permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np + 0.5, 0, 1)
            input_images.append(img_np)

            gt_mag = blur_field[0, 2].cpu().numpy()
            pred_mag = out[0, 2].cpu().numpy()

            gt_mags.append(gt_mag)
            pred_mags.append(pred_mag)

    # Save input, gt, and pred magnitude grids
    display_magnitude_grid(input_images, [f"Input {i}" for i in range(5)],
                           os.path.join(output_dir, "input_grid.png"), show_legend=False)
    display_magnitude_grid(gt_mags, [f"GT {i}" for i in range(5)],
                           os.path.join(output_dir, "gt_mag_grid.png"), show_legend=True)
    display_magnitude_grid(pred_mags, [f"Pred {i}" for i in range(5)],
                           os.path.join(output_dir, "pred_mag_grid.png"), show_legend=True)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_validation_subset(args.val_dir, args.crop_size)

    if args.model_name.lower() == 'random':
        model = RandomModel()
        print("Using RandomModel")
    elif args.model_name.lower() == 'zero':
        model = ZeroModel()
        print("Using ZeroModel")
    else:
        model = build_MIMOUnet_net(args.model_name)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Using MIMO-UNetPlus")

    model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.is_dir:
        visualize_direction_samples(model, dataset, device, args.output_dir)
    else:
        visualize_magnitude_samples(model, dataset, device, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, required=True, help='Validation data path')
    parser.add_argument('--model_path', type=str, default='', help='Path to the trained model .pth file')
    parser.add_argument('--model_name', type=str, default='MIMO-UNetPlus', help='Model name')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save output image')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size (must match training)')
    parser.add_argument('--is_dir', action='store_true', help='Use direction-aware visualization if set')
    args = parser.parse_args()

    main(args)
