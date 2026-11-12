"""One-page PDF screening report (matplotlib, no extra dependencies)."""
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def screening_pdf(result, images, out_path, patient_code):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.07, 0.95, "OmniDx Vision — AI-generated screening result", fontsize=15, weight="bold")
    demo = "   [DEMO MODEL — NOT FOR CLINICAL USE]" if result["model"].get("demo") else ""
    fig.text(0.07, 0.925, f"Patient: {patient_code}    Generated: {datetime.now():%d %b %Y %H:%M}{demo}",
             fontsize=9, color="#34444F")

    for i, (key, title) in enumerate([("processed", "Processed image"), ("segmentation", "Region of interest"),
                                      ("heatmap", "Grad-CAM heatmap")]):
        ax = fig.add_axes([0.06 + i * 0.30, 0.62, 0.28, 0.28])
        ax.imshow(images[key])
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    probs = "\n".join(f"    {c}: {p:.3f}" for c, p in result["probabilities"].items())
    share = result["explainability"]["heatmap_share_in_roi"]
    share_text = "n/a" if share is None else f"{share:.2f}"
    risk = result.get("risk")
    risk_text = "not available" if not risk else f"{risk['risk_category']} ({risk['risk_score']:.2f}) — {risk['note']}"
    lines = [
        f"Module: {result['module_name']}",
        f"Model: {result['model']['architecture']}  (version {result['model']['version']})",
        f"Model output: {result['top_class']}   confidence {result['confidence']:.3f}",
        f"Class probabilities:\n{probs}",
        f"Heatmap share inside region of interest: {share_text}",
        f"Risk assessment (Random Forest): {risk_text}",
        "Status: awaiting clinician review",
        f"Note: {result.get('label_note') or ''}",
    ]
    if result.get("warnings"):
        lines.append("Image warnings: " + "; ".join(result["warnings"]))
    fig.text(0.07, 0.58, "\n\n".join(textwrap.fill(l, 110, replace_whitespace=False) for l in lines),
             fontsize=9.5, va="top")
    fig.text(0.07, 0.05, textwrap.fill(result["disclaimer"], 110), fontsize=8.5, color="#6E3504")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
