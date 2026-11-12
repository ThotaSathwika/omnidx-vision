"""Generate a small SYNTHETIC chest-X-ray-like dataset to smoke-test OmniDx end to end.

These are drawn shapes, not medical images. Usage:
    python scripts/make_demo_data.py --out datasets/demo_xray --patients 60
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def draw(rng, opacity, size=256):
    img = np.full((size, size), 25, np.float32)
    c = size // 2
    cv2.ellipse(img, (c, c + 10), (int(size * .40), int(size * .45)), 0, 0, 360, 170, -1)   # body
    for dx in (-1, 1):                                                                     # lungs
        cv2.ellipse(img, (c + dx * int(size * .17) + int(rng.integers(-4, 5)), c),
                    (int(size * .12), int(size * .28)), 0, 0, 360, 60, -1)
    for y in range(int(size * .25), int(size * .75), 18):                                  # ribs
        cv2.line(img, (int(size * .2), y), (int(size * .8), y + 10), 95, 2)
    if opacity:
        side = rng.choice([-1, 1])
        mask = np.zeros_like(img)
        cv2.circle(mask, (c + side * int(size * .17), c + int(rng.integers(-30, 40))),
                   int(rng.integers(14, 26)), 1, -1)
        img += cv2.GaussianBlur(mask, (31, 31), 0) * 85
    img = cv2.GaussianBlur(img, (5, 5), 0) + rng.normal(0, 8, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.putText(img, "L", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, 230, 2)                  # side marker
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/demo_xray")
    ap.add_argument("--patients", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for p in range(a.patients):
        label = "opacity" if p % 2 else "normal"
        age = int(rng.integers(25, 80))
        spo2 = float(np.clip(rng.normal(92 if label == "opacity" else 96, 2), 85, 100))
        for k in range(2):  # two images per patient -> tests patient-level splitting
            name = f"P{p:03d}_{k}.png"
            d = Path(a.out) / label
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / name), draw(rng, label == "opacity"))
            rows.append({"filename": name, "patient_id": f"P{p:03d}", "age": age, "spo2": round(spo2, 1)})
    pd.DataFrame(rows).to_csv(Path(a.out) / "metadata.csv", index=False)
    print(f"Wrote {len(rows)} synthetic images to {a.out}")


if __name__ == "__main__":
    main()
