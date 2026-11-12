"""Dataset scanning, patient-level splitting, leakage checks and the PyTorch Dataset.

Supported layouts under datasets/<module>/:
  A) train/<class>/*.png, val/<class>/*, test/<class>/*   (Kaggle pre-split; val optional)
  B) <class>/*.png                                         (we split 70/15/15)
Optional metadata.csv with columns: filename, patient_id, and any clinical features.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .preprocessing import load_image, preprocess

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def scan(root):
    root = Path(root)
    rows = []
    pre_split = any((root / s).is_dir() for s in SPLITS)
    for split_dir in ([root / s for s in SPLITS if (root / s).is_dir()] if pre_split else [root]):
        for cls in sorted(d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith(".")):
            for p in sorted(cls.rglob("*")):
                if p.suffix.lower() in EXTS:
                    rows.append({"path": str(p), "filename": p.name, "label": cls.name,
                                 "split": split_dir.name if pre_split else None})
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")
    df = pd.DataFrame(rows)
    meta = root / "metadata.csv"
    if meta.exists():
        df = df.merge(pd.read_csv(meta), on="filename", how="left")
    if "patient_id" not in df or df["patient_id"].isna().all():
        df["patient_id"] = df["path"]  # no grouping info: each image is its own group
    df["patient_id"] = df["patient_id"].fillna(df["path"]).astype(str)
    return df


def _group_split(df, frac, seed):
    """Split rows so that no patient appears on both sides."""
    if df["patient_id"].nunique() == len(df):
        a, b = train_test_split(df.index, test_size=frac, stratify=df["label"], random_state=seed)
        return df.loc[a], df.loc[b]
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    a, b = next(gss.split(df, groups=df["patient_id"]))
    return df.iloc[a], df.iloc[b]


def make_splits(df, val=0.15, test=0.15, seed=42):
    df = df.copy()
    if df["split"].isna().all():
        rest, te = _group_split(df, test, seed)
        tr, va = _group_split(rest, val / (1 - test), seed)
        df.loc[tr.index, "split"], df.loc[va.index, "split"], df.loc[te.index, "split"] = "train", "val", "test"
    elif "val" not in set(df["split"]):  # Kaggle sets often ship without a usable val folder
        tr = df[df["split"] == "train"]
        _, va = _group_split(tr, val, seed)
        df.loc[va.index, "split"] = "val"
    return df.reset_index(drop=True)


def dhash(path, size=16):
    """Perceptual difference-hash (256 bits) of an image."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def near_duplicates(bits, splits, max_dist=12, chunk=2048):
    """Indices of images whose near-duplicate (Hamming distance <= max_dist of 256 bits)
    sits in a different split. Distances come from a chunked matrix product."""
    s = np.where(np.asarray(bits), 1.0, -1.0).astype(np.float32)
    split_ids = np.unique(splits, return_inverse=True)[1]
    leaked = set()
    for start in range(0, len(s), chunk):
        dist = (s.shape[1] - s[start:start + chunk] @ s.T) / 2
        i, j = np.where((dist <= max_dist) & (split_ids[start:start + chunk, None] != split_ids[None, :]))
        leaked.update((i + start).tolist())
        leaked.update(j.tolist())
    return sorted(leaked)


def leakage_report(df):
    """Patients or near-duplicate images shared between train/val/test."""
    df = df.reset_index(drop=True)
    bits = np.stack([dhash(p) for p in df["path"]])
    dup = near_duplicates(bits, df["split"].tolist())
    df["leaked_duplicate"] = df.index.isin(dup)
    by = df.groupby("patient_id")["split"].nunique()
    shared = by[by > 1].index
    report = {"patients_in_multiple_splits": int(len(shared)),
              "near_duplicate_images_across_splits": int(len(dup))}
    report["ok"] = not shared.size and not dup
    return report, df


def drop_leaked_duplicates(df):
    """Keep the train copy of a cross-split duplicate; drop its val/test copies."""
    bad = df["leaked_duplicate"] & (df["split"] != "train")
    return df[~bad].reset_index(drop=True), int(bad.sum())


def augmentations(modality):
    """Train-time augmentation that stays anatomically plausible for each modality."""
    common = [T.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.95, 1.05)),
              T.ColorJitter(brightness=0.15, contrast=0.15)]
    if modality == "dermoscopy":  # lesions have no fixed orientation
        return T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(90)] + common)
    if modality == "mri":
        return T.Compose([T.RandomHorizontalFlip()] + common)
    return T.Compose(common)  # no flips for chest X-ray / abdomen: left-right anatomy matters


def to_tensor(rgb):
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return T.Normalize(MEAN, STD)(x)


class OmniDataset(Dataset):
    """Loads an image, runs the modality pipeline (cached on disk) and augments if training."""

    def __init__(self, df, classes, modality, size, train=False, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.idx = {c: i for i, c in enumerate(classes)}
        self.modality, self.size = modality, size
        self.aug = augmentations(modality) if train else None
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def processed(self, i):
        path = self.df.loc[i, "path"]
        if self.cache:
            key = hashlib.md5(f"{path}|{self.modality}|{self.size}".encode()).hexdigest()
            f = self.cache / f"{key}.png"
            if f.exists():
                return cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        out = preprocess(load_image(path), self.modality, self.size)
        if self.cache:
            cv2.imwrite(str(f), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return out

    def __getitem__(self, i):
        x = torch.from_numpy(self.processed(i)).permute(2, 0, 1)
        if self.aug:
            x = self.aug(x)
        x = T.Normalize(MEAN, STD)(x.float() / 255.0)
        return x, self.idx[self.df.loc[i, "label"]]
