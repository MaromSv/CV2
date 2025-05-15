# DPT_blur: Pixel-Wise Blur Quantification using Dense Prediction Transformers

This directory contains code for **DPT_blur**, a model adapted from the Dense Prediction Transformer (DPT) architecture, originally designed for tasks like depth estimation and segmentation. Here, the model has been repurposed to **predict pixel-wise blur vectors** from a single input blurred image. The goal is to quantify the motion blur at each pixel, potentially estimating both its **magnitude** and **orientation**.

## Directory Structure & Key Scripts

*   `dpt_lib/`: Contains the core implementation of the modified DPT model architecture, derived from a subset of the original DPT repository.
*   `weights/`: Intended location for storing pre-trained model weights.
*   `train_dpt_blur.py`: Script used for training the DPT_blur model. It handles data loading, model optimization, and saving checkpoints.
*   `run_dpt_blur_quantification.py`: Script to run inference using a trained DPT_blur model on input images to generate blur maps/vector fields.
*   `visualize_blur_map.py`: Utility script for visualizing the predicted blur information (e.g., as heatmaps or quiver plots).
*   `results/`: Default directory for saving output blur maps or visualizations.
*   `showcase_dpt_blur.ipynb`: (To be created) A Jupyter Notebook demonstrating how to load the model, process a test image, and visualize the predicted blur.

## Integration Context

This repo provides a model and method to *predict* blur characteristics from images, whereas `ID-Blau` focuses on *generating* synthetically blurred images with ground truth blur information for training.

## Setup

1.  **Dependencies:** Ensure you have PyTorch, OpenCV (`opencv-python`), NumPy, Matplotlib, and tqdm installed.
    ```bash
    pip install torch torchvision opencv-python numpy matplotlib tqdm
    ```
2.  **Weights:** Download the pre-trained DPT weights required for backbone initialization. For example, using `dpt_large`:
    *   Download `dpt_large-ade20k-b12dca68.pt` (e.g., from the original DPT repository releases or other sources).
    *   Place the downloaded `.pt` file inside the `DPT_blur/weights/` directory.
    *   *(Note: This `weights` directory is ignored by git)*.

## Ground Truth Data (for Training)

The `train_dpt_blur.py` script expects the training and validation datasets to be structured as follows:

```
<your_base_dataset_directory>/    # e.g., dataset_DPT_blur/
├── train/
│   ├── blur/                   # Contains .png, .jpg, or .jpeg blurred images
│   └── condition/              # Contains corresponding .npy ground truth maps
└── val/
    ├── blur/                   # Contains .png, .jpg, or .jpeg blurred images
    └── condition/              # Contains corresponding .npy ground truth maps
```

*   **Image and GT Pairing:** Each image in a `blur/` directory (e.g., `train/blur/image_001.png`) must have a corresponding ground truth file with the same base name in the respective `condition/` directory (e.g., `train/condition/image_001.npy`).
*   **GT Format:** Each `.npy` file should contain a NumPy array of shape `(3, H, W)`, representing the blur vector components `(bx, by, magnitude)` for each pixel.

### Data Processing During Training and Validation:

*   **Training (`is_train=True`):**
    *   **Random Cropping:** If a `crop_size` is specified (typically matching `args.img_size` passed to the training script), random patches of this size are extracted from the training images and their corresponding ground truth maps.
    *   **Random Horizontal Flip:** If enabled (default is True in `train_dpt_blur.py`), images and their GT maps are randomly flipped horizontally. The `bx` component of the GT map is appropriately inverted during this flip.
*   **Validation (`is_train=False`):**
    *   **Center Cropping:** If a `crop_size` is specified (now matches `args.img_size` by default for validation in `train_dpt_blur.py`), center patches of this size are extracted from the validation images and their corresponding ground truth maps.
    *   **No Random Flip:** Random flipping is disabled during validation.

This strategy ensures that the model is validated on data that is processed similarly to the training data (in terms of patch size), providing a fairer assessment of learning.

## Usage Examples

*(Ensure you are in the main `CV2_code` directory if running commands as written below, or adjust paths accordingly if running from `CV2_code/CV2/DPT_blur/`)*

1.  **Restructure your ID-Blau (or similar) Dataset (if needed):**
    (Assuming `restructure_dataset.py` is in `CV2/DPT_blur/` and your raw dataset is in `dataset/GOPRO_Large_Reblur/` and you want the output in `dataset_DPT_blur/`)
    ```bash
    python CV2/DPT_blur/restructure_dataset.py dataset/GOPRO_Large_Reblur/ dataset_DPT_blur/
    ```
    Move or ensure `dataset_DPT_blur/` is accessible, for example at `CV2/DPT_blur/data/dataset_DPT_blur/`.

2.  **Inspect Dataset Labels (Optional):**
    (To check statistics of your `.npy` ground truth files)
    ```bash
    python CV2/DPT_blur/inspect_labels.py
    ```

3.  **Start Training:**
    The training script now expects a single base directory for the dataset.
    ```bash
    # Example: running from CV2_code directory
    # Assumes your restructured dataset is at CV2/DPT_blur/data/dataset_DPT_blur/
    python CV2/DPT_blur/train_dpt_blur.py \
        --dataset_dir CV2/DPT_blur/data/dataset_DPT_blur/ \
        --weights CV2/DPT_blur/weights/dpt_large-ade20k-b12dca68.pt \
        --model_type dpt_large \
        --img_size 384 \
        --epochs 50 \
        --batch_size 2 \
        --lr 0.0001 \
        --num_workers 4 \
        --checkpoint_dir CV2/DPT_blur/checkpoints/
    ```

4.  **Resume Training:**
    ```bash
    # Example: running from CV2_code directory
    python CV2/DPT_blur/train_dpt_blur.py \
        --dataset_dir CV2/DPT_blur/data/dataset_DPT_blur/ \
        --weights CV2/DPT_blur/weights/dpt_large-ade20k-b12dca68.pt \
        --model_type dpt_large \
        --img_size 384 \
        --epochs 50 \
        --batch_size 2 \
        --lr 0.0001 \
        --num_workers 4 \
        --checkpoint_dir CV2/DPT_blur/checkpoints/ \
        --resume CV2/DPT_blur/checkpoints/dpt_blur_latest.pth
    ```

5.  **Run Inference (Example):**
    (Assuming the script `run_dpt_blur_quantification.py` exists and is set up)
    ```bash
    # Example: running from CV2_code directory
    python CV2/DPT_blur/run_dpt_blur_quantification.py \
        --model_path CV2/DPT_blur/checkpoints/dpt_blur_best.pth \
        --input_path /path/to/your/test_image.png \
        --output_path CV2/DPT_blur/results/predicted_blur_map.npy \
        --model_type dpt_large 
    ```

6.  **Visualize Output (Example):**
    (Assuming `visualize_blur_map.py` exists and is set up)
    ```bash
    # Example: running from CV2_code directory
    python CV2/DPT_blur/visualize_blur_map.py CV2/DPT_blur/results/predicted_blur_map.npy --step 20
    ``` 