"""Load spatially aligned RGB imagery, morphology rasters and binary labels."""

import random
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

MORPHOLOGY_BANDS = ("density", "area", "nnd", "height", "footprint")
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def index_rasters(directory, suffix):
    """Pair city_ID rasters by identity, retaining their actual file paths."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    pattern = re.compile(rf"^(.+?)[_-](\d+)[_-]{suffix}$", re.IGNORECASE)
    indexed = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".tif", ".tiff"):
            continue
        match = pattern.fullmatch(path.stem)
        if not match:
            raise ValueError(f"Expected city_001_{suffix}.tif, received {path.name}")
        sample_id = f"{match.group(1).lower()}_{match.group(2)}"
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample: {sample_id}")
        indexed[sample_id] = path
    if not indexed:
        raise ValueError(f"No GeoTIFFs found in {directory}")
    return indexed


def read_rasters(image_path, building_path, label_path=None, input_shape=512, label_encoding="01"):
    """Read prepared rasters without implicit resizing or reprojection."""
    with rasterio.open(image_path) as image, rasterio.open(building_path) as building:
        if image.count < 3 or building.count != 5:
            raise ValueError("Expected >=3 RGB bands and exactly 5 building bands")
        if any(dtype != "uint8" for dtype in image.dtypes[:3]):
            raise ValueError("RGB input must be uint8")
        if image.crs is None or building.crs != image.crs:
            raise ValueError("Paired rasters must have the same defined CRS")
        if image.shape != building.shape or not image.transform.almost_equals(building.transform, 1e-9):
            raise ValueError("Image and building rasters must share one pixel grid")
        if image.shape != (input_shape, input_shape):
            raise ValueError(f"Expected {input_shape}x{input_shape} patches")
        rgb = image.read([1, 2, 3])
        morphology = building.read().astype(np.float32)
        profile = image.profile.copy()
    if not np.isfinite(morphology).all() or morphology.min() < -1e-6 or morphology.max() > 1 + 1e-6:
        raise ValueError("Building morphology bands must be finite and pre-normalized to [0,1]")
    if not np.isin(morphology[4], [0, 1]).all():
        raise ValueError("Band 5 must contain binary building footprints")
    label = None
    if label_path is not None:
        with rasterio.open(label_path) as source:
            if (source.count != 1 or source.shape != (input_shape, input_shape)
                    or source.crs != profile["crs"]
                    or not source.transform.almost_equals(profile["transform"], 1e-9)):
                raise ValueError("Label and image rasters must share one pixel grid")
            label = source.read(1)
        if label_encoding == "0255":
            if not np.isin(label, [0, 255]).all():
                raise ValueError("0255 encoding requires 0=background and 255=urban village")
            label = (label == 255).astype(np.int64)
        elif label_encoding == "01":
            if not np.isin(label, [0, 1, 255]).all():
                raise ValueError("01 encoding requires 0=background, 1=urban village, 255=ignore")
            label = label.astype(np.int64)
        else:
            raise ValueError("label_encoding must be 01 or 0255")
    return rgb, morphology, label


class UVSegmentationDataset(Dataset):
    """Aligned patch triplets under split/{rs,building,label}/.

    Geometric augmentation is synchronized across inputs and labels; color
    augmentation affects only RGB imagery. For inference set with_labels=False.
    """

    def __init__(self, dataset_dir, split="train", input_shape=512, training=False,
                 label_encoding="01", with_labels=True):
        directory = Path(dataset_dir) / split
        image_index = index_rasters(directory / "rs", "rs")
        building_index = index_rasters(directory / "building", "building")
        indices = {"image_path": image_index, "building_path": building_index}
        if with_labels:
            indices["label_path"] = index_rasters(directory / "label", "label")
        for name, indexed in indices.items():
            if set(indexed) != set(image_index):
                raise ValueError(f"Incomplete or unmatched samples in {name}")
        self.samples = [{"sample_id": sample_id,
                         **{key: indexed[sample_id] for key, indexed in indices.items()}}
                        for sample_id in sorted(image_index)]
        self.input_shape = input_shape
        self.training = training
        self.label_encoding = label_encoding
        self.color_jitter = ColorJitter(0.2, 0.2, 0.2, 0.05)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        rgb, morphology, label = read_rasters(
            sample["image_path"], sample["building_path"], sample.get("label_path"),
            self.input_shape, self.label_encoding,
        )
        image = TF.to_pil_image(np.moveaxis(rgb, 0, -1))
        building = torch.from_numpy(morphology)
        label = torch.from_numpy(label) if label is not None else None
        if self.training:
            for dimension, flip in ((-1, TF.hflip), (-2, TF.vflip)):
                if random.random() > 0.5:
                    image, building = flip(image), building.flip([dimension])
                    if label is not None:
                        label = label.flip([dimension])
            if random.random() > 0.5:
                k = random.choice((1, 2, 3))
                image, building = TF.rotate(image, 90 * k), torch.rot90(building, k, (-2, -1))
                if label is not None:
                    label = torch.rot90(label, k, (-2, -1))
            if random.random() > 0.5:
                image = self.color_jitter(image)
        data = {
            "image": TF.normalize(TF.to_tensor(image), RGB_MEAN, RGB_STD),
            "building": building.contiguous(), "sample_id": sample["sample_id"],
            "image_path": str(sample["image_path"]),
        }
        if label is not None:
            data["label"] = label.contiguous()
        return data
