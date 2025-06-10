import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import Compose

# Import DPT model and transforms
try:
    from dpt_lib.models import DPT
    from dpt_lib.transforms import Resize, NormalizeImage, PrepareForNet
    from dpt_lib.blocks import Interpolate
except ImportError as e:
    print("Error: Could not import local DPT library (dpt_lib). Make sure it exists in the same directory as the script.")

# Import dataset and model utilities
from data_loader import BlurMapDataset
from model_utils import create_dpt_blur_model

# --- Define collate_fn at the top level ---
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch) 