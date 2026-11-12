import json

import numpy as np
import pytest
import torch

from omnidx.evaluate import compute_metrics, ece
from omnidx.fusion import risk_category
from omnidx.gradcam import GradCAM, roi_energy
from omnidx.inference import ModelNotDeployed, Predictor
from omnidx.models import OmniNet
from omnidx.monitoring import assess_metric, compare_exams
from omnidx.preprocessing import load_image


def test_metrics_known_values():
    y = np.array([0, 0, 0, 1, 1])
    probs = np.array([[.9, .1], [.8, .2], [.4, .6], [.3, .7], [.6, .4]])
    m = compute_metrics(y, probs, ["normal", "disease"])
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["per_class"]["disease"]["recall_sensitivity"] == pytest.approx(0.5)
    assert m["per_class"]["disease"]["specificity"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == [[2, 1], [1, 1]]
    assert 0 <= m["roc_auc"] <= 1 and len(m["accuracy_95ci"]) == 2


def test_ece_perfect_calibration_is_zero():
    assert ece(np.array([0, 1]), np.array([[1.0, 0.0], [0.0, 1.0]])) == pytest.approx(0)


@pytest.mark.parametrize("arch", ["resnet18", "densenet121", "efficientnet_b0", "mobilenet_v3_large"])
def test_gradcam_every_architecture(arch):
    model = OmniNet(arch, 3, pretrained=False)
    cam, probs, idx = GradCAM(model)(torch.rand(1, 3, 96, 96))
    assert cam.shape == (96, 96) and 0 <= cam.min() and cam.max() <= 1
    assert probs.sum() == pytest.approx(1, abs=1e-4) and 0 <= idx < 3
    assert model.embed(torch.rand(2, 3, 96, 96)).shape == (2, model.embed_dim)


def test_roi_energy():
    cam = np.zeros((10, 10)); cam[:5] = 1
    mask = np.zeros((10, 10), np.uint8); mask[:5] = 255
    assert roi_energy(cam, mask) == pytest.approx(1)
    assert roi_energy(cam, None) is None


def test_risk_category():
    t = {"low": 0.33, "high": 0.66}
    assert [risk_category(p, t) for p in (0.1, 0.5, 0.9)] == ["Low", "Moderate", "High"]


def test_omnicare_rules():
    assert assess_metric([96, 95, 91], 94, 100)["status"] == "Change detected"
    assert assess_metric([1.0, 1.05])["status"] == "Stable"
    assert "consult your healthcare professional" in assess_metric([1.0, 1.5])["message"]


def test_untrained_module_is_reported_not_faked():
    with pytest.raises(ModelNotDeployed):
        Predictor("untrained_demo")


def test_training_outputs(trained_run):
    for f in ("model.pt", "metrics.json", "splits.csv", "leakage.json", "confusion_matrix.png", "roc_curve.png",
              "fusion.joblib", "fusion_metrics.json", "fusion_importance.png"):
        assert (trained_run / f).exists(), f
    m = json.loads((trained_run / "metrics.json").read_text())
    assert m["accuracy"] >= 0.7  # synthetic task is easy; a regression here means training broke
    assert m["xai_audit"]["images"] > 0
    fm = json.loads((trained_run / "fusion_metrics.json").read_text())
    assert {"cnn_only", "radiomics_only", "fusion", "clinical_only"} <= set(fm["ablation"])


def test_prediction_and_change_monitoring(trained_run, demo_data):
    p = Predictor("demo_xray")
    r1, i1 = p.predict(load_image(next(demo_data.glob("normal/*.png"))), {"age": 50, "spo2": 97})
    r2, i2 = p.predict(load_image(next(demo_data.glob("opacity/*.png"))), {"age": 50, "spo2": 90})
    assert r2["result_type"] == "AI-generated screening result" and r2["status"] == "awaiting_clinician_review"
    assert sum(r2["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert r2["risk"]["risk_category"] in ("Low", "Moderate", "High")
    assert i2["heatmap"].shape == i2["processed"].shape

    rules = {"prob_review": .15, "prob_change": .05, "area_review": .2, "diff_change": 12}
    same, _ = compare_exams({**r1, **i1}, {**r1, **i1}, rules, ["opacity"])
    assert same["status"] == "Stable"
    if r2["top_class"] == "opacity" and r1["top_class"] == "normal":
        worse, _ = compare_exams({**r1, **i1}, {**r2, **i2}, rules, ["opacity"])
        assert worse["status"] == "Requires clinical review"
        assert "relapse" not in worse["message"].lower()
