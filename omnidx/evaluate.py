"""Model evaluation: per-class sensitivity/specificity, ROC-AUC, confusion matrix,
bootstrap confidence intervals, calibration (ECE) and plots.

Re-evaluate a saved model:
    python -m omnidx.evaluate --run models/pneumonia/<version>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)


@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def ece(y, probs, bins=10):
    """Expected Calibration Error: gap between confidence and actual accuracy."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def bootstrap_ci(y, pred, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(fn(y[i], pred[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def compute_metrics(y, probs, classes):
    pred = probs.argmax(1)
    labels = list(range(len(classes)))
    cm = confusion_matrix(y, pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = {}
    for i, c in enumerate(classes):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        per_class[c] = {"precision": float(prec[i]), "recall_sensitivity": float(rec[i]),
                        "specificity": float(tn / (tn + fp)) if tn + fp else None,
                        "f1": float(f1[i]), "support": int(support[i])}
    macro_f1 = lambda a, b: f1_score(a, b, average="macro", labels=labels, zero_division=0)
    out = {
        "n_test": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "accuracy_95ci": bootstrap_ci(y, pred, accuracy_score),
        "macro_f1": float(macro_f1(y, pred)),
        "macro_f1_95ci": bootstrap_ci(y, pred, macro_f1),
        "ece": ece(y, probs),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
    try:
        if len(classes) == 2:
            out["roc_auc"] = float(roc_auc_score(y, probs[:, 1]))
        else:
            out["roc_auc_macro_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", labels=labels))
    except ValueError:
        out["roc_auc"] = None  # only one class present in the test set
    return out


def plot(y, probs, classes, out_dir):
    out_dir = Path(out_dir)
    cm = confusion_matrix(y, probs.argmax(1), labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(4 + len(classes) * .4, 3.6 + len(classes) * .4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    for i, c in enumerate(classes):
        if 0 < (y == i).sum() < len(y):
            fpr, tpr, _ = roc_curve(y == i, probs[:, i])
            ax.plot(fpr, tpr, label=c)
        if len(classes) == 2:
            break
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def xai_audit(model, ds, segment_kind, n=50):
    """Mean share of Grad-CAM energy inside the anatomical ROI over n test images."""
    from .gradcam import GradCAM, roi_energy
    from .segmentation import segment
    if not segment_kind:
        return None
    cam_fn, vals = GradCAM(model), []
    for i in range(min(n, len(ds))):
        rgb = ds.processed(i)
        cam, _, _ = cam_fn(ds[i][0][None])
        v = roi_energy(cam, segment(rgb, segment_kind))
        if v is not None:
            vals.append(v)
    return {"images": len(vals), "mean_heatmap_energy_in_roi": float(np.mean(vals)) if vals else None}


def main():
    import pandas as pd
    from torch.utils.data import DataLoader
    from .data import OmniDataset
    from .models import load_checkpoint
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="models/<module>/<version>")
    a = ap.parse_args()
    run = Path(a.run)
    model, meta = load_checkpoint(run / "model.pt")
    df = pd.read_csv(run / "splits.csv")
    ds = OmniDataset(df[df.split == "test"], meta["classes"], meta["modality"], meta["image_size"])
    y, p = predict(model, DataLoader(ds, batch_size=32))
    m = compute_metrics(y, p, meta["classes"])
    plot(y, p, meta["classes"], run)
    print(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
