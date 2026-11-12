import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from omnidx.data import leakage_report, make_splits, scan
from omnidx.preprocessing import ImageValidationError, load_image, preprocess, validate
from omnidx.radiomics import features
from omnidx.segmentation import UNet, dice, iou, lung_mask, segment, train_unet


@pytest.mark.parametrize("modality", ["xray", "mri", "ct", "ultrasound", "dermoscopy"])
def test_every_modality_pipeline(demo_data, modality):
    img = load_image(next(demo_data.glob("normal/*.png")))
    out, steps = preprocess(img, modality, 160, keep_steps=True)
    assert out.shape == (160, 160, 3) and out.dtype == np.uint8
    assert steps[0][0] == "original" and len(steps) >= 4


def test_validation_rejects_bad_images():
    with pytest.raises(ImageValidationError):
        validate(np.zeros((32, 32, 3), np.uint8), "xray")
    with pytest.raises(ImageValidationError):
        validate(np.full((128, 128, 3), 90, np.uint8), "xray")
    with pytest.raises(ImageValidationError):
        load_image(b"not an image")


def test_lung_mask_finds_two_lungs(demo_data):
    img = preprocess(load_image(next(demo_data.glob("normal/*.png"))), "xray", 224)
    mask = lung_mask(img)
    n, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
    assert n - 1 == 2 and 0.1 < (mask > 0).mean() < 0.4
    assert segment(img, None) is None


def test_dice_iou():
    a = np.zeros((10, 10)); a[:5] = 1
    b = np.zeros((10, 10)); b[:5, :5] = 1
    assert dice(a, a) == pytest.approx(1)
    assert iou(a, b) == pytest.approx(0.5)
    assert dice(a, b) == pytest.approx(2 * 25 / 75)


def test_unet_shapes_and_training_step():
    x = torch.rand(4, 3, 64, 64)
    y = (torch.rand(4, 1, 64, 64) > 0.5).float()
    assert UNet()(x).shape == (4, 1, 64, 64)
    assert isinstance(train_unet(x, y, epochs=1), UNet)


def test_radiomics(demo_data):
    img = preprocess(load_image(next(demo_data.glob("opacity/*.png"))), "xray", 224)
    f = features(img, lung_mask(img))
    assert {"glcm_contrast", "fo_entropy", "lbp_0", "shape_area_ratio"} <= set(f)
    assert all(np.isfinite(v) for v in f.values())


def test_patient_level_split_and_leakage(demo_data, tmp_path):
    df = make_splits(scan(demo_data))
    assert set(df.split) == {"train", "val", "test"}
    assert df.groupby("patient_id").split.nunique().max() == 1  # no patient in two splits
    report, _ = leakage_report(df)
    assert report["ok"]

    # plant a resized copy of a training image in the test split
    row = df[df.split == "train"].iloc[0]
    dup = tmp_path / "dup.png"
    cv2.imwrite(str(dup), cv2.resize(cv2.imread(row.path), (300, 300)))
    extra = pd.DataFrame([{**row.to_dict(), "path": str(dup), "split": "test", "patient_id": "X"}])
    report, _ = leakage_report(pd.concat([df, extra], ignore_index=True))
    assert report["near_duplicate_images_across_splits"] == 2 and not report["ok"]
