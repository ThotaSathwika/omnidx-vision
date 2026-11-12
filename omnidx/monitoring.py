"""Longitudinal monitoring: previous-vs-current scan comparison and OmniCare trends.

The wording never confirms relapse or recurrence; it flags change for clinical review.
"""
import cv2
import numpy as np

MESSAGES = {
    "Requires clinical review": "Potentially significant change detected compared with the previous examination. "
                                "Clinical review is recommended.",
    "Change detected": "Change detected in image-derived indicators compared with the previous examination. "
                       "Clinical review is recommended.",
    "Improvement indicator": "Image-derived indicators have moved toward the normal pattern compared with the "
                             "previous examination. This does not confirm recovery; clinical review is recommended.",
    "Stable": "No significant change detected in image-derived indicators compared with the previous examination.",
}
OMNICARE_ALERT = "Change detected in monitored health data. Please consult your healthcare professional."


def align(prev_rgb, curr_rgb):
    """Register the current scan onto the previous one (affine ECC) so pixels line up."""
    a = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    b = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
        ok = True
    except cv2.error:
        ok = False
    h, w = a.shape
    aligned = cv2.warpAffine(curr_rgb, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, ok


def compare_exams(prev, curr, rules, positive_classes):
    """prev/curr: dicts with positive_probability, top_class, roi_area_ratio, processed (RGB), mask."""
    dp = curr["positive_probability"] - prev["positive_probability"]
    area_change = None
    if prev.get("roi_area_ratio") and curr.get("roi_area_ratio") is not None:
        area_change = (curr["roi_area_ratio"] - prev["roi_area_ratio"]) / prev["roi_area_ratio"]

    aligned, ok = align(prev["processed"], curr["processed"])
    diff = cv2.absdiff(cv2.cvtColor(prev["processed"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY))
    roi = prev.get("mask")
    diff_score = float(diff[roi > 0].mean() if roi is not None and (roi > 0).any() else diff.mean())

    became_positive = curr["top_class"] in positive_classes and prev["top_class"] not in positive_classes
    if became_positive or dp >= rules["prob_review"] or (area_change or 0) >= rules["area_review"]:
        status = "Requires clinical review"
    elif dp <= -rules["prob_review"]:
        status = "Improvement indicator"
    elif abs(dp) >= rules["prob_change"] or diff_score >= rules["diff_change"]:
        status = "Change detected"
    else:
        status = "Stable"
    heat = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_INFERNO)
    return {"status": status, "message": MESSAGES[status],
            "indicators": {"probability_change": round(dp, 4),
                           "roi_area_change": None if area_change is None else round(area_change, 4),
                           "aligned_pixel_difference": round(diff_score, 2), "registration_converged": ok}
            }, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def assess_metric(values, ref_low=None, ref_high=None, rel_change=0.2):
    """OmniCare trend for one clinician-selected parameter. values: list of numbers, oldest first."""
    if not values:
        return {"status": "No data"}
    latest, baseline = values[-1], values[0]
    change = (latest - baseline) / abs(baseline) if baseline else 0.0
    out_of_range = (ref_low is not None and latest < ref_low) or (ref_high is not None and latest > ref_high)
    flagged = out_of_range or (len(values) > 1 and abs(change) >= rel_change)
    return {"status": "Change detected" if flagged else "Stable", "latest": latest, "baseline": baseline,
            "change_from_baseline": round(change, 4), "outside_reference_range": bool(out_of_range),
            "message": OMNICARE_ALERT if flagged else None}
