"""One-epoch training and evaluation routines."""

import torch
from tqdm import tqdm

from .utils_metrics import SegmentationMetrics


def fit_one_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, max_grad_norm=1.0):
    training = optimizer is not None
    model.train(training)
    metric = SegmentationMetrics()
    total_loss, sample_count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="Training" if training else "Evaluating", leave=False):
            images = batch["image"].to(device)
            buildings = batch["building"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, buildings)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * len(images)
            sample_count += len(images)
            metric.update(logits.detach().argmax(1).cpu().numpy(), labels.cpu().numpy())
    if sample_count == 0:
        raise ValueError("Data loader is empty")
    return {"loss": total_loss / sample_count, **metric.compute()}, metric.confusion_matrix
