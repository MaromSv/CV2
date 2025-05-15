# MIMO-UNet for Blur Field Prediction

This repository contains the implementation of MIMO-UNet (Multi-Input Multi-Output UNet) adapted for blur field prediction. The model predicts pixel-wise blur vectors from a single input blurred image, quantifying both the magnitude and direction of motion blur at each pixel.

## Directory Structure

```
MIMO_UNet/
├── MIMOUNet.py              # Core model architecture
├── train_blur_field.py      # Training script
├── test_blur_field.py       # Testing and inference script
├── blur_losses.py           # Custom loss functions
├── blur_utils.py            # Utility functions
├── dataloader.py            # Data loading utilities
├── config/
│   └── config.yaml          # Configuration parameters
└── README.md                # This file
```

## Model Architecture

MIMO-UNet is a multi-scale encoder-decoder network that processes images at multiple scales and produces outputs at multiple resolutions. For blur field prediction, we've adapted it to output 3-channel tensors representing:

- Channel 0: Horizontal blur component (bx)
- Channel 1: Vertical blur component (by)
- Channel 2: Blur magnitude

### Model Specifications

- **Input dimensions**: 3×H×W (RGB image)
- **Output dimensions**: 3×H×W (blur field components)
- **Base channels**: 64 (configurable)
- **Model variants**: 
  - MIMO-UNet: Standard version
  - MIMO-UNetPlus: Enhanced version with better performance

## Dataset Format

The training dataset should be structured as follows:

```
dataset_dir/
├── train/
│   ├── blur/                # Contains blurred images (.png, .jpg)
│   └── condition/           # Contains blur field ground truth (.npy)
└── val/
    ├── blur/                # Contains blurred images (.png, .jpg)
    └── condition/           # Contains blur field ground truth (.npy)
```

Each `.npy` file should contain a NumPy array of shape `(3, H, W)`, representing the blur vector components `(bx, by, magnitude)` for each pixel.

## Training

### Prerequisites

- PyTorch 1.7+
- CUDA-capable GPU with 8+ GB memory
- NumPy, OpenCV, Matplotlib, tqdm

### Training Command

```bash
python train_blur_field.py \
    --train_dir /path/to/dataset/train \
    --val_dir /path/to/dataset/val \
    --output_dir ./experiments/blur_field \
    --model_name MIMO-UNetPlus \
    --batch_size 8 \
    --epochs 300 \
    --lr 1e-4 \
    --crop_size 256 \
    --base_channels 64
```

### Key Training Parameters

- **batch_size**: Batch size for training (default: 8)
- **epochs**: Number of training epochs (default: 300)
- **lr**: Learning rate (default: 1e-4)
- **crop_size**: Size of random crops during training (default: 256)
- **base_channels**: Base number of channels in MIMO-UNet (default: 64)
- **model_name**: Model variant to use (MIMO-UNet or MIMO-UNetPlus)

### Loss Function

The model is trained using a combination of:
- Charbonnier loss for overall accuracy
- Directional loss (cosine similarity) for vector direction
- Magnitude loss for blur strength

## Testing and Inference

### Batch Testing

```bash
python test_blur_field.py \
    --test_dir /path/to/test/dataset \
    --model_path /path/to/model.pth \
    --output_dir ./results \
    --model_name MIMO-UNetPlus
```

### Single Image Inference

```bash
python test_blur_field.py \
    --single_image /path/to/image.png \
    --model_path /path/to/model.pth \
    --output_dir ./results \
    --model_name MIMO-UNetPlus
```

## Visualization

The test script generates visualizations of the predicted blur fields:
- Quiver plots showing blur direction and magnitude
- Heatmaps of blur magnitude
- Combined visualizations with the original image

## Performance

When properly trained, the model achieves:
- High accuracy in predicting blur direction
- Precise estimation of blur magnitude
- Effective handling of complex, spatially-varying blur

## Citation

If you use this code in your research, please cite the original MIMO-UNet paper:
```
@inproceedings{cho2021rethinking,
  title={Rethinking coarse-to-fine approach in single image deblurring},
  author={Cho, Sung Jin and Ji, Seo Woon and Hong, Jun-Pyo and Jung, Seung-Won and Ko, Sung-Jea},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4641--4650},
  year={2021}
}
```
