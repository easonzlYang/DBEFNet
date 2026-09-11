"""Shared device, checkpoint and raster-output helpers."""

import json
import os
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image

from .checkpoint import normalize_state_dict


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu")
    return torch.device(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def load_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint
    for key in ("model", "model_state_dict", "state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    model.load_state_dict(normalize_state_dict(state), strict=True)


def save_checkpoint(model, optimizer, scheduler, epoch, args, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "args": vars(args)}, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_mask(prediction, reference_path, output_path, overlay_path=None):
    """Save a georeferenced 0/1 mask and an optional RGB overlay."""
    with rasterio.open(reference_path) as reference:
        if prediction.shape != reference.shape:
            raise ValueError("Prediction shape differs from the reference grid")
        profile = dict(driver="GTiff", height=reference.height, width=reference.width,
                       transform=reference.transform, crs=reference.crs, count=1,
                       dtype="uint8", nodata=255, compress="lzw")
        rgb = np.moveaxis(reference.read([1, 2, 3]), 0, -1) if overlay_path else None
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(prediction.astype(np.uint8), 1)
        destination.set_band_description(1, "0=background; 1=urban village; 255=nodata")
    if overlay_path:
        rgb = rgb.astype(np.float32)
        rgb[prediction == 1] = 0.55 * rgb[prediction == 1] + 0.45 * np.array([220, 40, 40])
        Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(overlay_path)
