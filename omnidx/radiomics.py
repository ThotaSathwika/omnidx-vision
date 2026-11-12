"""Hand-crafted (radiomics-style) image features computed inside the ROI.

These interpretable texture/shape features feed the Random Forest alongside CNN features.
"""
import cv2
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops


def features(img, mask=None):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    roi = mask > 0 if mask is not None and (mask > 0).sum() > 50 else np.ones_like(g, bool)
    px = g[roi].astype(np.float32)

    hist = np.bincount(px.astype(np.uint8), minlength=256) / px.size
    f = {
        "fo_mean": float(px.mean()), "fo_std": float(px.std()),
        "fo_skew": float(stats.skew(px)), "fo_kurtosis": float(stats.kurtosis(px)),
        "fo_entropy": float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum()),
        "fo_p10": float(np.percentile(px, 10)), "fo_p90": float(np.percentile(px, 90)),
    }

    # GLCM texture on the ROI bounding box, quantised to 32 grey levels
    ys, xs = np.where(roi)
    crop = (g[ys.min():ys.max() + 1, xs.min():xs.max() + 1] // 8).astype(np.uint8)
    glcm = graycomatrix(crop, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                        symmetric=True, normed=True)
    for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        f[f"glcm_{p}"] = float(graycoprops(glcm, p).mean())

    lbp = local_binary_pattern(g, 8, 1, "uniform")[roi]
    for i, v in enumerate(np.bincount(lbp.astype(int), minlength=10)[:10] / lbp.size):
        f[f"lbp_{i}"] = float(v)

    if mask is not None and (mask > 0).sum() > 50:
        props = max(regionprops(label(mask > 0)), key=lambda r: r.area)
        f.update(shape_area_ratio=float((mask > 0).mean()), shape_eccentricity=float(props.eccentricity),
                 shape_solidity=float(props.solidity))
    else:
        f.update(shape_area_ratio=0.0, shape_eccentricity=0.0, shape_solidity=0.0)
    return f
