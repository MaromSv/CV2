# DPT_blur: Pixel-Wise Blur Quantification using Dense Prediction Transformers

This directory contains code for **DPT_blur**, a model adapted from the Dense Prediction Transformer (DPT) architecture, originally designed for tasks like depth estimation. Here, the model has been repurposed to **predict pixel-wise blur vectors** from a single input blurred image.

The goal is to quantify the motion blur at each pixel, potentially estimating both its **magnitude** and **direction**. This information can be valuable for various computer vision tasks, including blur removal, understanding scene dynamics, or synthesizing realistic blur effects.

## Directory Structure & Key Scripts

*   `dpt_lib/`: Contains the core implementation of the modified DPT model architecture.
*   `weights/`: Intended location for storing pre-trained model weights.
*   `train_dpt_blur.py`: Script used for training the DPT_blur model. It likely handles data loading, model optimization, and saving checkpoints.
*   `run_dpt_blur_quantification.py`: Script to run inference using a trained DPT_blur model on input images to generate blur maps/vector fields.
*   `visualize_blur_map.py`: Utility script for visualizing the predicted blur information (e.g., as heatmaps or quiver plots).
*   `results/`: Default directory for saving output blur maps or visualizations.
*   `showcase_dpt_blur.ipynb`: (To be created) A Jupyter Notebook demonstrating how to load the model, process a test image, and visualize the predicted blur.

## Integration Context

This module is part of the larger `CV2` repository. It complements other efforts, such as the `ID-Blau` dataset generation, by providing a method to *predict* blur characteristics from images, whereas `ID-Blau` focuses on *generating* synthetically blurred images with ground truth blur information for training.

## Usage Example

Please refer to the `showcase_dpt_blur.ipynb` notebook in this directory for a practical example of loading the model and running it on test images.

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

The `train_dpt_blur.py` script expects ground truth data to be provided as `.npy` files.

*   **Format:** Each `.npy` file should correspond to a training image and contain a NumPy array of shape `(2, H, W)`.
*   **Representation:** The script currently includes a placeholder to convert data stored as `(Magnitude, Angle_in_Radians)` in the `.npy` file into the `(bx, by)` format needed for training. 
    *   **IMPORTANT:** If your ground truth `.npy` files *already* store the `(bx, by)` components directly, you **MUST** edit `train_dpt_blur.py` and remove/comment out the conversion block within the `BlurMapDataset.__getitem__` method (clearly marked with `TODO` comments).

## Usage Examples

*(Ensure you are in the main `CV2` directory)*

1.  **Run Inference (Example):**
    ```bash
    python DPT_blur/run_dpt_blur_quantification.py \
        --weights DPT_blur/weights/dpt_large-ade20k-b12dca68.pt \
        --model_type dpt_large \
        --output_file DPT_blur/outputs/example_blur_vector.pt \
        --print_shapes
    ```
    *(You might need to `mkdir DPT_blur/outputs` first)*

2.  **Visualize Output:**
    ```bash
    python DPT_blur/visualize_blur_map.py DPT_blur/outputs/example_blur_vector.pt --step 20
    ```

3.  **Start Training:**
    ```bash
    # TODO: Update with actual paths to your prepared datasets
    python DPT_blur/train_dpt_blur.py \
        --blurred_dir_train /path/to/your/train/blurred_images \
        --gt_dir_train /path/to/your/train/gt_maps_npy \
        --blurred_dir_val /path/to/your/validation/blurred_images \
        --gt_dir_val /path/to/your/validation/gt_maps_npy \
        --weights DPT_blur/weights/dpt_large-ade20k-b12dca68.pt \
        --model_type dpt_large \
        --epochs 50 \
        --batch_size 4 \
        --lr 0.0001 \
        --checkpoint_dir DPT_blur/checkpoints
    ```

4.  **Resume Training:**
    ```bash
    # TODO: Update paths
    python DPT_blur/train_dpt_blur.py \
        --blurred_dir_train /path/to/your/train/blurred_images \
        --gt_dir_train /path/to/your/train/gt_maps_npy \
        # ... (other args as above) ...
        --resume DPT_blur/checkpoints/dpt_blur_latest.pth 
    ``` 