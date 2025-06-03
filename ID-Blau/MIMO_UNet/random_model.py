#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from PIL import Image
from tqdm import tqdm

# --- path setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)      # ID-Blau
grandparent = os.path.dirname(parent_dir)        # CODE
sys.path.append(parent_dir)
sys.path.append(grandparent)

# --- imports from DPT_blur ---
try:
    from DPT_blur.data_loader import BlurMapDataset
    from DPT_blur.visualize_blur_map import visualize_multiple_blur_fields
except ImportError as e:
    print(f"Error importing DPT_blur modules: {e}")
    sys.exit(1)

# --- losses ---
from blur_losses import MultiScaleBlurFieldLoss

# --- number of samples to display ---
DISPLAY_SAMPLES = 5

# --- random‐model that returns multi‐scale random maps ---
class RandomModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B,3,H,W]
        B, C, H, W = x.shape
        outputs = []
        for scale in (0.25, 0.5, 1.0):
            h = int(H * scale)
            w = int(W * scale)
            # bx, by in [-0.5,0.5], mag in [0,1]
            bx  = torch.rand(B, 1, h, w, device=x.device, dtype=x.dtype) - 0.5
            by  = torch.rand(B, 1, h, w, device=x.device, dtype=x.dtype) - 0.5
            mag = torch.rand(B, 1, h, w, device=x.device, dtype=x.dtype)
            outputs.append(torch.cat([bx, by, mag], dim=1))
        return outputs

# --- helper to save component visualizations ---
def _save_components(tensor3ch, out_path):
    import matplotlib.pyplot as plt
    bx, by, mag = tensor3ch.cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title, cmap, vmin, vmax in [
        (axs[0], bx,  'bx',       'coolwarm', -1,   1),
        (axs[1], by,  'by',       'coolwarm', -1,   1),
        (axs[2], mag, 'magnitude','viridis',   0,   1),
    ]:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# --- utility to save exactly DISPLAY_SAMPLES and produce grids ---
def save_validation_grid(model, val_loader, output_dir, device):
    base     = os.path.join(output_dir, 'validation_samples')
    rand_dir = os.path.join(base, 'random_model')
    gt_dir   = os.path.join(base, 'gt')
    os.makedirs(rand_dir, exist_ok=True)
    os.makedirs(gt_dir,   exist_ok=True)

    pred_tensors = []
    gt_tensors   = []
    image_paths  = []

    for i, batch in enumerate(val_loader):
        if i >= DISPLAY_SAMPLES:
            break

        if isinstance(batch, dict):
            img = batch['blur'].to(device)
            gt  = batch['blur_field'].to(device)
        else:
            img, gt = batch[0].to(device), batch[1].to(device)

        # normalize input to [0..1]
        arr = img[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        arr = np.clip(arr, 0, 1)

        blur_path = os.path.join(gt_dir, f'sample_{i}_blur.png')
        Image.fromarray((arr * 255).astype(np.uint8)).save(blur_path)
        image_paths.append(blur_path)

        gt_tensor = gt[0].cpu()
        gt_pt     = os.path.join(gt_dir, f'sample_{i}_gt.pt')
        torch.save(gt_tensor, gt_pt)
        gt_tensors.append(gt_tensor)
        _save_components(gt_tensor,
                         os.path.join(gt_dir, f'sample_{i}_components.png'))

        # run random‐model and grab full-res
        outs      = model(img)
        pred_full = outs[-1][0].cpu()
        pred_tensors.append(pred_full)

        pred_pt = os.path.join(rand_dir, f'sample_{i}_pred.pt')
        torch.save(pred_full, pred_pt)
        _save_components(pred_full,
                         os.path.join(rand_dir, f'sample_{i}_components.png'))

    # build 5-column grids
    pred_grid = os.path.join(rand_dir, 'predictions_grid.png')
    visualize_multiple_blur_fields(pred_tensors, image_paths, pred_grid)

    gt_grid = os.path.join(gt_dir, 'gt_grid.png')
    visualize_multiple_blur_fields(gt_tensors, image_paths, gt_grid)

    return pred_grid, gt_grid

# --- main evaluation ---
def evaluate_random_model(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'random_model.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # dataset dirs
    blur_dir = os.path.join(args.val_dir, 'blur')
    cond_dir = os.path.join(args.val_dir, 'condition')
    if not (os.path.isdir(blur_dir) and os.path.isdir(cond_dir)):
        base = os.path.dirname(args.val_dir)
        alt  = os.path.join(base, 'condition')
        if os.path.isdir(alt):
            cond_dir = alt

    ds = BlurMapDataset(
        blurred_dir=blur_dir,
        gt_dir=cond_dir,
        transform=None,
        is_train=False,
        crop_size=256
    )

    # split 50/50
    n  = len(ds)
    v  = n // 2
    t  = n - v
    val_ds, test_ds = random_split(ds, [v, t],
                                   generator=torch.Generator().manual_seed(42))

    val_loader  = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model     = RandomModel().to(device)
    criterion = MultiScaleBlurFieldLoss()
    logging.info("Created random model that predicts uniform random blur fields.")

    def run_eval(loader, name):
        total_loss = 0.0
        total_mse  = 0.0
        total_psnr = 0.0
        model.eval()

        for batch in tqdm(loader, desc=name):
            img, gt = (batch['blur'], batch['blur_field']) if isinstance(batch, dict) else (batch[0], batch[1])
            img, gt = img.to(device), gt.to(device)

            # forward
            outs = model(img)
            loss, _ = criterion(outs, gt)

            # MSE/PSNR on full-res
            pred  = outs[-1]
            mse   = ((pred - gt)**2).mean().item()
            psnr  = 10 * np.log10(1.0 / mse) if mse > 0 else 0.0

            total_loss += loss.item()
            total_mse  += mse
            total_psnr += psnr

        L = len(loader)
        logging.info(f"{name} → Loss: {total_loss/L:.6f}, "
                     f"MSE: {total_mse/L:.6f}, PSNR: {total_psnr/L:.2f} dB")

    # Evaluate validation, save grids, then test
    run_eval(val_loader, 'Validation')
    pred_grid, gt_grid = save_validation_grid(model, val_loader, args.output_dir, device)
    logging.info(f"Saved validation grids:\n  PRED: {pred_grid}\n  GT:   {gt_grid}")
    run_eval(test_loader, 'Test')
    logging.info("Random model evaluation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Random Model for BlurField")
    parser.add_argument('--val_dir',     type=str, required=True,
                        help="root folder containing /blur and /condition")
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    evaluate_random_model(args)
