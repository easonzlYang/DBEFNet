"""Convenient DBEFNet checkpoint loading and batch prediction."""

import torch

from nets.DBEFNet import DBEFNet
from utils.utils import get_device, load_weights


class DBEFNetPredictor:
    def __init__(self, checkpoint, device="auto"):
        self.device = get_device(device)
        self.net = DBEFNet()
        load_weights(self.net, checkpoint)
        self.net = self.net.to(self.device).eval()

    @torch.inference_mode()
    def predict_batch(self, images, buildings):
        """Accept normalized RGB tensors and prepared morphology tensors."""
        logits = self.net(images.to(self.device), buildings.to(self.device))
        return logits.argmax(dim=1).cpu().numpy()
