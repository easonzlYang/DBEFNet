"""Predict urban village masks from paired image/building patches."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from dbefnet import DBEFNetPredictor
from utils.dataloader import UVSegmentationDataset
from utils.utils import save_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="datasets/CUGUV")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_shape", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_dir", default="outputs/predictions")
    parser.add_argument("--save_overlay", action="store_true")
    args = parser.parse_args()
    dataset = UVSegmentationDataset(args.dataset_dir, args.split, args.input_shape, with_labels=False)
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"{sample['sample_id']}_pred.tif").exists() for sample in dataset.samples):
        raise FileExistsError("Predictions already exist; choose another --save_dir")
    predictor = DBEFNetPredictor(args.checkpoint, args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    for batch in tqdm(loader, desc="Predicting"):
        predictions = predictor.predict_batch(batch["image"], batch["building"])
        for mask, sample_id, image_path in zip(predictions, batch["sample_id"], batch["image_path"]):
            overlay_path = output / f"{sample_id}_overlay.png" if args.save_overlay else None
            save_mask(mask, image_path, output / f"{sample_id}_pred.tif", overlay_path)
    print(f"Saved {len(dataset)} predictions to {output}")


if __name__ == "__main__":
    main()
