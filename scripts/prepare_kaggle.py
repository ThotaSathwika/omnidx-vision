"""Turn downloaded Kaggle datasets into the OmniDx layout (class folders + metadata.csv).

    python scripts/prepare_kaggle.py pneumonia   <download_dir> datasets/pneumonia
    python scripts/prepare_kaggle.py brain_tumor <download_dir> datasets/brain_tumor
    python scripts/prepare_kaggle.py skin_cancer <download_dir> datasets/skin_cancer

The datasets' own train/test folders are merged on purpose: OmniDx re-splits them by
patient (or lesion), because the original splits can put the same patient in both.
"""
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

IMG = {".png", ".jpg", ".jpeg"}


def images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMG and "__MACOSX" not in p.parts]


def copy_all(rows, out):
    """rows: dicts with src, label, filename and any metadata columns."""
    out = Path(out)
    for r in rows:
        d = out / r["label"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["src"], d / r["filename"])
    meta = pd.DataFrame(rows).drop(columns=["src", "label"])
    meta.to_csv(out / "metadata.csv", index=False)
    print(f"{len(rows)} images -> {out}")
    print(pd.Series([r["label"] for r in rows]).value_counts().to_string())


def pneumonia(src, out):
    """Chest X-Ray Images (Pneumonia), Kaggle: paultimothymooney/chest-xray-pneumonia.
    Patient IDs come from file names: person123_bacteria_4.jpeg -> person123, IM-0115-0001.jpeg -> IM-0115."""
    rows, seen = [], set()
    for p in images(src):
        label = p.parent.name.lower()
        if label not in ("normal", "pneumonia"):
            continue
        m = re.match(r"(person\d+)", p.stem) or re.match(r"(.*IM-\d+)", p.stem)
        name = f"{p.parent.parent.name}_{p.name}"
        if name in seen:  # the dataset ships some folders twice (chest_xray/chest_xray)
            continue
        seen.add(name)
        rows.append({"src": p, "label": label, "filename": name, "patient_id": m.group(1) if m else p.stem})
    copy_all(rows, out)


def brain_tumor(src, out):
    """Brain Tumor MRI Dataset, Kaggle: masoudnickparvar/brain-tumor-mri-dataset.
    No patient IDs are given, so train with --drop-duplicates to remove repeated scans."""
    classes = {"glioma", "meningioma", "notumor", "pituitary"}
    rows = [{"src": p, "label": p.parent.name.lower(), "filename": f"{p.parent.parent.name}_{p.name}"}
            for p in images(src) if p.parent.name.lower() in classes]
    copy_all(rows, out)


def skin_cancer(src, out):
    """Skin Cancer MNIST: HAM10000, Kaggle: kmader/skin-cancer-mnist-ham10000.
    malignant = melanoma, basal cell carcinoma, actinic keratosis/intraepithelial carcinoma.
    Images of the same lesion share a lesion_id, used for grouped splitting; age is a clinical feature."""
    meta = pd.read_csv(next(Path(src).rglob("HAM10000_metadata.csv")))
    paths = {p.stem: p for p in images(src) if p.stem.startswith("ISIC_")}
    rows = []
    for r in meta.itertuples():
        if r.image_id in paths:
            rows.append({"src": paths[r.image_id], "filename": f"{r.image_id}.jpg",
                         "label": "malignant" if r.dx in ("mel", "bcc", "akiec") else "benign",
                         "patient_id": r.lesion_id, "age": r.age, "dx": r.dx})
    copy_all(rows, out)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("pneumonia", "brain_tumor", "skin_cancer"):
        sys.exit(__doc__)
    globals()[sys.argv[1]](sys.argv[2], sys.argv[3])
