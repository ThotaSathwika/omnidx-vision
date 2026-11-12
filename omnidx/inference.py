"""Run a deployed disease module on one image.

There are no hard-coded predictions: if a module has no trained model, ModelNotDeployed
is raised and the API reports that honestly.
"""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np

from .config import DISCLAIMER, MODELS_DIR, get_module, load_config
from .data import to_tensor
from .fusion import image_features, predict_risk
from .gradcam import GradCAM, hot_area, overlay, roi_energy
from .models import load_checkpoint
from .preprocessing import preprocess, validate
from .segmentation import segment


class ModelNotDeployed(RuntimeError):
    pass


def active_version(module, models_dir=None):
    f = Path(models_dir or MODELS_DIR) / module / "ACTIVE"
    return f.read_text().strip() if f.exists() else None


def list_versions(module, models_dir=None):
    d = Path(models_dir or MODELS_DIR) / module
    return sorted(p.name for p in d.iterdir() if (p / "model.pt").exists()) if d.exists() else []


def run_summary(module, version, models_dir=None):
    """Metrics recorded at training time (never invented)."""
    d = Path(models_dir or MODELS_DIR) / module / version
    load = lambda n: json.loads((d / n).read_text()) if (d / n).exists() else None
    return {"version": version, "meta": load("meta.json"), "metrics": load("metrics.json"),
            "fusion": load("fusion_metrics.json"), "leakage": load("leakage.json")}


class Predictor:
    def __init__(self, module, version=None, cfg=None, models_dir=None):
        self.mod = get_module(module, cfg or load_config())
        self.version = version or active_version(module, models_dir)
        run = Path(models_dir or MODELS_DIR) / module / str(self.version)
        if not self.version or not (run / "model.pt").exists():
            raise ModelNotDeployed(f"No trained model is deployed for '{module}'. "
                                   f"Train one with: python -m omnidx.train --module {module} --data datasets/{module}")
        self.model, self.meta = load_checkpoint(run / "model.pt")
        self.cam = GradCAM(self.model)
        self.fusion = joblib.load(run / "fusion.joblib") if (run / "fusion.joblib").exists() else None

    def predict(self, img, clinical=None):
        m, mod = self.meta, self.mod
        warnings = validate(img, m["modality"])
        processed = preprocess(img, m["modality"], m["image_size"])
        mask = segment(processed, mod.get("segmentation"))
        cam, probs, idx = self.cam(to_tensor(processed)[None])
        p_pos = float(sum(p for c, p in zip(m["classes"], probs) if c in m["positive_class"]))

        seg_view = processed.copy()
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(seg_view, contours, -1, (0, 220, 120), 2)

        result = {
            "module": m["module"], "module_name": mod["name"], "label_note": mod.get("label_note"),
            "model": {"architecture": m["architecture"], "version": m["version"], "demo": m.get("demo", False)},
            "result_type": "AI-generated screening result",
            "classes": m["classes"],
            "probabilities": {c: round(float(p), 4) for c, p in zip(m["classes"], probs)},
            "top_class": m["classes"][idx], "confidence": round(float(probs[idx]), 4),
            "positive_probability": round(p_pos, 4),
            "explainability": {"method": "Grad-CAM", "heatmap_share_in_roi": roi_energy(cam, mask),
                               "heatmap_hot_area": hot_area(cam)},
            "roi_area_ratio": float((mask > 0).mean()) if mask is not None else None,
            "risk": None, "warnings": warnings,
            "status": "awaiting_clinician_review", "disclaimer": DISCLAIMER,
        }
        if self.fusion:
            feats = image_features(self.model, processed, m["classes"], mod.get("segmentation"))
            result["risk"] = predict_risk(self.fusion, feats, clinical)
        images = {"original": img, "processed": processed, "segmentation": seg_view,
                  "heatmap": overlay(processed, cam), "mask": mask}
        return result, images


@lru_cache(maxsize=16)
def get_predictor(module, version=None):
    return Predictor(module, version)
