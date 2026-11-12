"""Train, validate and test one disease module.

    python -m omnidx.train --module pneumonia --data datasets/pneumonia --epochs 15

Creates models/<module>/<version>/ with model.pt, splits.csv, leakage.json, history.json,
metrics.json, confusion_matrix.png and roc_curve.png, and marks the version ACTIVE.
"""
import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import evaluate as ev
from .config import MODELS_DIR, get_module, load_config
from .data import OmniDataset, drop_leaked_duplicates, leakage_report, make_splits, scan
from .models import OmniNet, save_checkpoint


def run_epoch(model, loader, loss_fn, opt=None, device="cpu"):
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(y)
            n += len(y)
    return total / n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config")
    ap.add_argument("--arch", help="override the registry architecture")
    ap.add_argument("--size", type=int, help="override the registry image size")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-epochs", type=int, default=1, help="epochs training only the head")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--drop-duplicates", action="store_true", help="drop val/test copies of train images")
    ap.add_argument("--fusion", action="store_true", help="also train the Random Forest layer")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = load_config(a.config)
    mod = get_module(a.module, cfg)
    arch, size = a.arch or mod["architecture"], a.size or mod["image_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. data, patient-level split and leakage check
    df = make_splits(scan(a.data), seed=a.seed)
    report, df = leakage_report(df)
    if report["patients_in_multiple_splits"]:
        raise SystemExit(f"Leakage: {report['patients_in_multiple_splits']} patients appear in more than one "
                         "split. Remove the dataset's own split folders so OmniDx can split by patient.")
    if report["near_duplicate_images_across_splits"]:
        if not a.drop_duplicates:
            raise SystemExit(f"Leakage: {report['near_duplicate_images_across_splits']} near-duplicate images "
                             "cross splits. Re-run with --drop-duplicates to remove the val/test copies.")
        df, dropped = drop_leaked_duplicates(df)
        report["dropped_from_val_test"] = dropped
    classes = sorted(df["label"].unique())
    missing = [c for c in mod["positive_class"] if c not in classes]
    if missing:
        print(f"WARNING: positive_class {missing} not among dataset classes {classes}")

    cache = Path(a.data) / ".cache"
    sets = {s: OmniDataset(df[df.split == s], classes, mod["modality"], size, train=(s == "train"), cache_dir=cache)
            for s in ("train", "val", "test")}
    loaders = {s: DataLoader(d, batch_size=a.batch, shuffle=(s == "train"), num_workers=a.workers)
               for s, d in sets.items()}
    counts = {s: d.df["label"].value_counts().to_dict() for s, d in sets.items()}
    print("Images per split:", json.dumps(counts))

    # 2. model, class-weighted loss for imbalanced data
    model = OmniNet(arch, len(classes), pretrained=not a.no_pretrained).to(device)
    freq = sets["train"].df["label"].value_counts().reindex(classes).fillna(0).values + 1
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(freq.sum() / (len(freq) * freq), dtype=torch.float32).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    # 3. train with early stopping on validation macro-F1
    best, best_f1, wait, history = None, -1, 0, []
    for epoch in range(1, a.epochs + 1):
        model.freeze_backbone(epoch <= a.freeze_epochs)
        t = time.time()
        tr_loss = run_epoch(model, loaders["train"], loss_fn, opt, device)
        va_loss = run_epoch(model, loaders["val"], loss_fn, None, device)
        y, p = ev.predict(model, loaders["val"], device)
        f1 = ev.compute_metrics(y, p, classes)["macro_f1"]
        sched.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_macro_f1": f1})
        print(f"epoch {epoch:2d}  train_loss {tr_loss:.4f}  val_loss {va_loss:.4f}  val_macro_f1 {f1:.4f}  "
              f"({time.time() - t:.0f}s)")
        if f1 > best_f1:
            best, best_f1, wait = copy.deepcopy(model.state_dict()), f1, 0
        else:
            wait += 1
            if wait >= a.patience:
                print("Early stopping.")
                break
    model.load_state_dict(best)

    # 4. test once, on data never used for training or model selection
    y, p = ev.predict(model, loaders["test"], device)
    metrics = ev.compute_metrics(y, p, classes)
    metrics["xai_audit"] = ev.xai_audit(model.cpu(), sets["test"], mod.get("segmentation"))

    version = datetime.now().strftime("v%Y%m%d-%H%M%S")
    run = MODELS_DIR / a.module / version
    meta = {"module": a.module, "version": version, "architecture": arch, "classes": classes,
            "modality": mod["modality"], "image_size": size, "positive_class": mod["positive_class"],
            "trained_at": datetime.now().isoformat(timespec="seconds"), "pretrained": not a.no_pretrained,
            "split_counts": counts, "dataset": mod.get("dataset"), "demo": bool(mod.get("demo")),
            "data_root": str(Path(a.data).resolve()), "segmentation": mod.get("segmentation"),
            "label_note": mod.get("label_note")}
    save_checkpoint(run / "model.pt", model, meta)
    df.drop(columns=["hash"], errors="ignore").to_csv(run / "splits.csv", index=False)
    for name, obj in (("leakage", report), ("history", history), ("metrics", metrics), ("meta", meta)):
        (run / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))
    ev.plot(y, p, classes, run)
    (MODELS_DIR / a.module / "ACTIVE").write_text(version)
    print(f"\nTest accuracy {metrics['accuracy']:.4f} (95% CI {metrics['accuracy_95ci']}), "
          f"macro-F1 {metrics['macro_f1']:.4f}\nSaved {run}")

    if a.fusion:
        from .fusion import train_fusion
        train_fusion(run, mod, cfg)
    return run


if __name__ == "__main__":
    main()
