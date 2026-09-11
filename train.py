"""Train DBEFNet or evaluate a trained model on a specified split."""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from nets.DBEFNet import DBEFNet
from nets.dbefnet_training import CombinedLoss, polynomial_factor
from utils.dataloader import UVSegmentationDataset
from utils.utils import get_device, load_weights, save_checkpoint, seed_worker, set_seed, write_json
from utils.utils_fit import fit_one_epoch


def get_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    known, _ = config_parser.parse_known_args()
    defaults = {}
    if known.config:
        with Path(known.config).open(encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle) or {}
        if not isinstance(defaults, dict):
            raise ValueError("The configuration root must be a mapping")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML file containing command-line option defaults")
    parser.add_argument("--dataset_dir", default="datasets/CUGUV")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--input_shape", type=int, default=512)
    parser.add_argument("--label_encoding", choices=("01", "0255"), default="01")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--focal_alpha", type=float, default=0.75)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--weight_focal", type=float, default=1.0)
    parser.add_argument("--weight_dice", type=float, default=1.0)
    parser.add_argument("--weight_ce", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="outputs/DBEFNet")
    parser.add_argument("--eval_split", default="test", help="Dataset split used with --only_evaluate")
    parser.add_argument("--only_evaluate", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint to evaluate with --only_evaluate")
    unknown = set(defaults) - {action.dest for action in parser._actions}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    parser.set_defaults(**defaults)
    return parser.parse_args()


def main():
    args = get_args()
    if args.num_epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    set_seed(args.seed)
    device = get_device(args.device)
    common = dict(dataset_dir=args.dataset_dir, input_shape=args.input_shape, label_encoding=args.label_encoding)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    output = Path(args.output_dir)
    criterion = CombinedLoss(
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        weight_focal=args.weight_focal, weight_dice=args.weight_dice, weight_ce=args.weight_ce,
    )

    if args.only_evaluate:
        if not args.checkpoint:
            raise ValueError("--only_evaluate requires --checkpoint")
        dataset = UVSegmentationDataset(split=args.eval_split, **common)
        model = DBEFNet()
        load_weights(model, args.checkpoint)
        model = model.to(device)
        loader = DataLoader(dataset, shuffle=False, **loader_args)
        metrics, matrix = fit_one_epoch(model, loader, criterion, device)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "metrics.json", {
            "split": args.eval_split, "samples": len(dataset), "units": "fractions",
            "metrics": metrics, "confusion_matrix": matrix.tolist(),
        })
        print(json.dumps(metrics, indent=2))
        return

    if args.checkpoint:
        raise ValueError("--checkpoint is used with --only_evaluate")
    if args.batch_size < 2:
        raise ValueError("Training requires --batch_size >= 2")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Select an empty --output_dir for this training run")
    train_data = UVSegmentationDataset(split=args.train_split, training=True, **common)
    val_data = UVSegmentationDataset(split=args.val_split, **common)
    train_ids = {row["sample_id"] for row in train_data.samples}
    if train_ids & {row["sample_id"] for row in val_data.samples}:
        raise ValueError("Training and validation samples must be disjoint")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    if not len(train_loader):
        raise ValueError("Training data must contain at least one full batch")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", vars(args))
    model = DBEFNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: polynomial_factor(step, total_steps, args.poly_power))
    best_miou = -1.0
    patience_counter = 0
    history = []
    for epoch in range(1, args.num_epochs + 1):
        train_metrics, _ = fit_one_epoch(
            model, train_loader, criterion, device, optimizer, scheduler, args.max_grad_norm)
        val_metrics, _ = fit_one_epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
               "train_loss": train_metrics["loss"], "train_mIoU": train_metrics["mIoU"],
               "val_loss": val_metrics["loss"], "val_mIoU": val_metrics["mIoU"]}
        history.append(row)
        save_checkpoint(model, optimizer, scheduler, epoch, args, output / "last_checkpoint.pth")
        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            patience_counter = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, output / "best_checkpoint.pth")
        else:
            patience_counter += 1
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
        if patience_counter >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement")
            break


if __name__ == "__main__":
    main()
