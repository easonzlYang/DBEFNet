"""Training objective and polynomial learning-rate schedule."""

import torch
from torch import nn
from torch.nn import functional as F


class CombinedLoss(nn.Module):
    """Weighted sum of focal loss, Dice loss and cross entropy (equal weights by default).

    Focal alpha is a scalar multiplier. Dice is averaged over images and classes.
    Label 255 is ignored after the dataset's configured label conversion.
    """

    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, weight_focal=1.0,
                 weight_dice=1.0, weight_ce=1.0, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.weights = (weight_focal, weight_dice, weight_ce)
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, labels):
        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, labels, reduction="none", ignore_index=self.ignore_index)[valid]
        focal = (self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce).mean()
        one_hot = F.one_hot(labels.masked_fill(~valid, 0), num_classes=2).permute(0, 3, 1, 2)
        mask = valid.unsqueeze(1)
        probs, one_hot = logits.softmax(1) * mask, one_hot * mask
        intersection = (probs * one_hot).sum(dim=(2, 3))
        cardinality = (probs + one_hot).sum(dim=(2, 3))
        scores = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice = 1 - scores[valid.flatten(1).any(1)].mean()
        weight_focal, weight_dice, weight_ce = self.weights
        return weight_focal * focal + weight_dice * dice + weight_ce * ce.mean()


def polynomial_factor(step, total_steps, power=0.9):
    """Multiplier used after each optimizer step."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    return max(0.0, 1.0 - step / total_steps) ** power
