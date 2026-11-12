"""Grad-CAM explainability plus a simple audit of where the heatmap points."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping on the model's last feature block."""

    def __init__(self, model):
        self.model = model.eval()
        self.acts = None
        model.features.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out
        if out.requires_grad:  # skipped during normal no-grad inference
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        """x: 1x3xHxW tensor. Returns (cam HxW in [0,1], probabilities, class_idx)."""
        self.model.zero_grad()
        x = x.clone().requires_grad_(True)  # lets gradients flow even when the backbone is frozen
        logits = self.model(x)
        probs = F.softmax(logits, 1)[0].detach().cpu().numpy()
        idx = int(probs.argmax()) if class_idx is None else class_idx
        logits[0, idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # importance of each channel
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))       # weighted sum of feature maps
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, probs, idx


def overlay(rgb, cam, alpha=0.4):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def roi_energy(cam, mask):
    """Share of heatmap energy that falls inside the anatomical ROI (lungs, brain, lesion).
    Low values warn that the model may be using shortcuts such as text markers or borders."""
    if mask is None or (mask > 0).sum() == 0:
        return None
    return float(cam[mask > 0].sum() / (cam.sum() + 1e-8))


def hot_area(cam, thr=0.5):
    """Fraction of the image where the heatmap is above `thr`."""
    return float((cam >= thr).mean())
