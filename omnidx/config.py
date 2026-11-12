"""Loads the disease-module registry (configs/modules.yaml)."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("OMNIDX_MODELS", ROOT / "models"))

DISCLAIMER = ("OmniDx provides AI-assisted screening and clinical decision-support information. "
              "It is not a substitute for professional medical diagnosis or treatment. "
              "Results should be reviewed by a qualified healthcare professional.")
RISK_NOTE = "Risk assessment is not a diagnosis."


def load_config(path=None):
    path = Path(path or os.environ.get("OMNIDX_CONFIG", ROOT / "configs" / "modules.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, m in cfg["modules"].items():
        m["key"] = key
        pos = m.get("positive_class")
        m["positive_class"] = pos if isinstance(pos, list) else [pos]
    return cfg


def get_module(key, cfg=None):
    cfg = cfg or load_config()
    if key not in cfg["modules"]:
        raise KeyError(f"Unknown disease module '{key}'. Available: {list(cfg['modules'])}")
    return cfg["modules"][key]
