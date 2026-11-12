"""Region-of-interest segmentation.

Classical masks (lungs, brain, skin lesion) work without labelled masks and are used for
ROI-restricted features and to audit Grad-CAM. A compact U-Net is included for datasets
that do provide masks (e.g. tumour or liver masks), evaluated with Dice and IoU.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def _largest(mask, n=1):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(np.uint8)
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:n] + 1
    return np.isin(labels, order).astype(np.uint8)


def _clear_border(mask):
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    return (np.isin(labels, border, invert=True) & (labels > 0)).astype(np.uint8)


def lung_mask(img):
    """Lungs are the two large dark regions inside the body on a chest X-ray."""
    g = cv2.GaussianBlur(_gray(img), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = _largest(_clear_border(m), 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return ndimage.binary_fill_holes(m).astype(np.uint8) * 255


def brain_mask(img):
    g = cv2.GaussianBlur(_gray(img), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = _largest(m, 1)
    return ndimage.binary_fill_holes(m).astype(np.uint8) * 255


def lesion_mask(img):
    """Skin lesions are darker than surrounding skin; keep the component nearest the centre."""
    g = cv2.GaussianBlur(_gray(img), (7, 7), 0)
    _, m = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if count <= 1:
        return m * 255
    h, w = m.shape
    dist = [np.hypot(cx - w / 2, cy - h / 2) - np.sqrt(stats[i, cv2.CC_STAT_AREA])
            for i, (cx, cy) in enumerate(cents) if i > 0]
    keep = int(np.argmin(dist)) + 1
    return ndimage.binary_fill_holes(labels == keep).astype(np.uint8) * 255


CLASSICAL = {"lungs": lung_mask, "brain": brain_mask, "lesion": lesion_mask}


def segment(img, kind):
    """Return a 0/255 mask, or None when the module has no classical segmenter."""
    return CLASSICAL[kind](img) if kind in CLASSICAL else None


# ---------------- metrics ----------------

def dice(pred, target, eps=1e-6):
    p, t = np.asarray(pred) > 0, np.asarray(target) > 0
    return float((2 * (p & t).sum() + eps) / (p.sum() + t.sum() + eps))


def iou(pred, target, eps=1e-6):
    p, t = np.asarray(pred) > 0, np.asarray(target) > 0
    return float(((p & t).sum() + eps) / ((p | t).sum() + eps))


# ---------------- U-Net (for datasets with labelled masks) ----------------

def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.downs = nn.ModuleList([_block(in_ch, c[0]), _block(c[0], c[1]), _block(c[1], c[2])])
        self.mid = _block(c[2], c[3])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in (2, 1, 0)])
        self.decs = nn.ModuleList([_block(c[i] * 2, c[i]) for i in (2, 1, 0)])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = dec(torch.cat([up(x), skip], 1))
        return self.out(x)  # logits


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def train_unet(images, masks, epochs=20, lr=1e-3, device="cpu"):
    """images: float tensor N x 3 x H x W in [0,1]; masks: N x 1 x H x W in {0,1}."""
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for i in range(0, len(images), 8):
            x, y = images[i:i + 8].to(device), masks[i:i + 8].to(device)
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
