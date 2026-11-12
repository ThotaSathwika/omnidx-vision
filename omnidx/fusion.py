"""Random Forest risk layer that fuses CNN image features, radiomics and clinical data.

Trained on the VALIDATION split: the CNN never saw those images, so its outputs there are
honest (training-set CNN outputs are over-confident and would mislead the forest).
Evaluated on the untouched TEST split, with an ablation of each feature group.

    python -m omnidx.fusion --run models/copd/<version>
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from . import evaluate as ev
from .config import RISK_NOTE, load_config, get_module
from .data import OmniDataset, to_tensor
from .models import load_checkpoint
from .radiomics import features as radiomic_features
from .segmentation import segment


@torch.no_grad()
def image_features(model, rgb, classes, seg_kind):
    """CNN embedding + CNN probabilities + radiomics for one preprocessed image."""
    x = to_tensor(rgb)[None]
    emb = model.embed(x)
    probs = torch.softmax(model.head(emb), 1)[0].numpy()
    f = {f"cnn_emb_{i}": float(v) for i, v in enumerate(emb[0].numpy())}
    f.update({f"cnn_prob_{c}": float(p) for c, p in zip(classes, probs)})
    f.update({f"rad_{k}": v for k, v in radiomic_features(rgb, segment(rgb, seg_kind)).items()})
    return f


def build_table(model, meta, df, mod, cache):
    ds = OmniDataset(df, meta["classes"], meta["modality"], meta["image_size"], cache_dir=cache)
    rows = [image_features(model, ds.processed(i), meta["classes"], mod.get("segmentation")) for i in range(len(ds))]
    table = pd.DataFrame(rows, index=df.index)
    for c in mod.get("clinical_features", []):
        table[f"clin_{c}"] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    return table


def make_pipeline(cols, seed=0):
    emb = [c for c in cols if c.startswith("cnn_emb_")]
    rest = [c for c in cols if not c.startswith("cnn_emb_")]
    parts = [("rest", SimpleImputer(strategy="median", keep_empty_features=True), rest)] if rest else []
    if emb:
        parts.append(("emb", PCA(n_components=min(16, len(emb)), random_state=seed), emb))
    return Pipeline([("cols", ColumnTransformer(parts)),
                     ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                   class_weight="balanced", random_state=seed, n_jobs=-1))])


def risk_category(p, thresholds):
    return "Low" if p < thresholds["low"] else "High" if p >= thresholds["high"] else "Moderate"


def train_fusion(run, mod, cfg):
    run = Path(run)
    model, meta = load_checkpoint(run / "model.pt")
    df = pd.read_csv(run / "splits.csv")
    cache = Path(meta["data_root"]) / ".cache"
    val, test = df[df.split == "val"], df[df.split == "test"]
    if len(val) < 30:
        print(f"WARNING: only {len(val)} validation images to train the Random Forest; results will be noisy.")
    Xv, Xt = build_table(model, meta, val, mod, cache), build_table(model, meta, test, mod, cache)
    idx = {c: i for i, c in enumerate(meta["classes"])}
    yv, yt = val["label"].map(idx).values, test["label"].map(idx).values

    has_clinical = any(Xv[c].notna().any() for c in Xv if c.startswith("clin_"))
    groups = {"cnn_only": ("cnn_",), "radiomics_only": ("rad_",), "fusion": ("cnn_", "rad_", "clin_")}
    if has_clinical:
        groups["clinical_only"] = ("clin_",)
    results, pipes = {}, {}
    for name, prefixes in groups.items():
        cols = [c for c in Xv if c.startswith(prefixes) and (not c.startswith("clin_") or has_clinical)]
        pipe = make_pipeline(cols).fit(Xv[cols], yv)
        probs = pipe.predict_proba(Xt[cols])
        full = np.zeros((len(yt), len(meta["classes"])))
        full[:, pipe.classes_] = probs
        m = ev.compute_metrics(yt, full, meta["classes"])
        results[name] = {k: m.get(k) for k in ("accuracy", "accuracy_95ci", "macro_f1", "roc_auc",
                                               "roc_auc_macro_ovr", "ece")}
        pipes[name] = (pipe, cols)

    pipe, cols = pipes["fusion"]
    imp = permutation_importance(pipe, Xt[cols], yt, n_repeats=5, random_state=0, n_jobs=1)
    # embedding dimensions are summed into one "CNN features" bar to keep the chart readable
    scores = pd.Series(imp.importances_mean, index=cols)
    grouped = scores[~scores.index.str.startswith("cnn_emb_")]
    grouped["CNN embedding (all dims)"] = scores[scores.index.str.startswith("cnn_emb_")].sum()
    top = grouped.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top.index[::-1], top.values[::-1], color="#0B6E75")
    ax.set_title("Fusion Random Forest — permutation importance (test)")
    fig.tight_layout()
    fig.savefig(run / "fusion_importance.png", dpi=150)
    plt.close(fig)

    joblib.dump({"pipeline": pipe, "columns": cols, "classes": meta["classes"],
                 "positive_class": mod["positive_class"], "thresholds": cfg["risk_thresholds"]},
                run / "fusion.joblib")
    out = {"trained_on": "validation split", "n_train": int(len(yv)), "n_test": int(len(yt)),
           "clinical_features_available": has_clinical, "ablation": results,
           "top_features": {k: float(v) for k, v in top.items()}, "note": RISK_NOTE}
    (run / "fusion_metrics.json").write_text(json.dumps(out, indent=2))
    print("Random Forest ablation (test set):")
    for k, v in results.items():
        print(f"  {k:15s} accuracy {v['accuracy']:.3f}  macro-F1 {v['macro_f1']:.3f}")
    return out


def predict_risk(bundle, feats, clinical=None):
    """Risk category from the fused model. `clinical` maps feature name -> value."""
    row = dict(feats)
    for c in bundle["columns"]:
        if c.startswith("clin_"):
            row[c] = (clinical or {}).get(c[5:], np.nan)
    X = pd.DataFrame([row])[bundle["columns"]]
    probs = bundle["pipeline"].predict_proba(X)[0]
    classes = [bundle["classes"][i] for i in bundle["pipeline"].classes_]
    p_pos = float(sum(p for c, p in zip(classes, probs) if c in bundle["positive_class"]))
    return {"risk_score": round(p_pos, 4), "risk_category": risk_category(p_pos, bundle["thresholds"]),
            "note": RISK_NOTE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    meta = json.loads((Path(a.run) / "meta.json").read_text())
    train_fusion(a.run, get_module(meta["module"], cfg), cfg)


if __name__ == "__main__":
    main()
