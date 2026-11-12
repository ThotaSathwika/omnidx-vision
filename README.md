# OmniDx Vision: One Vision, Multiple Diseases, Smarter Decisions

OmniDx Vision is an explainable clinical decision-support platform for multi-disease medical image screening and longitudinal patient monitoring. Computer Vision is the primary domain and Machine Learning is the supporting interdisciplinary component.

> OmniDx provides AI-assisted screening and clinical decision-support information. It is not a substitute for professional medical diagnosis or treatment. Results should be reviewed by a qualified healthcare professional.

| Area | What is implemented | Where |
|---|---|---|
| **Image processing** | A separate pipeline for each modality. X-ray: median denoise, CLAHE, percentile normalisation. MRI: brain extraction. Ultrasound: speckle reduction. Dermoscopy: DullRazor hair removal and Shades-of-Gray colour constancy. Every pipeline also validates the image and resizes with aspect-preserving padding. | `omnidx/preprocessing.py` |
| **Segmentation** | Classical lung, brain and lesion masks (Otsu, morphology, connected components). A U-Net with Dice + BCE loss for datasets that have masks. Dice and IoU metrics. | `omnidx/segmentation.py` |
| **Deep learning** | ResNet18/50, DenseNet121, EfficientNet-B0 and MobileNetV3 with transfer learning. Class-weighted loss, modality-safe augmentation, head-only warm-up, cosine LR schedule, early stopping on validation macro-F1. | `omnidx/models.py`, `omnidx/train.py` |
| **Model evaluation** | Per-class sensitivity, specificity, precision and F1. ROC-AUC, confusion matrix, 95% bootstrap confidence intervals, calibration (ECE), ROC and confusion-matrix plots. | `omnidx/evaluate.py` |
| **Data integrity** | Patient-level splits (no patient in two splits). Perceptual-hash detection of near-duplicate images across train/val/test. Augmentation runs on training images only. | `omnidx/data.py` |
| **Explainable AI** | Grad-CAM for every architecture. An XAI audit measures how much heatmap energy falls inside the organ/lesion mask, which flags shortcut learning (text markers, borders). Permutation importance for the Random Forest. | `omnidx/gradcam.py` |
| **Random Forest** | Fuses CNN embedding (PCA), radiomics features (GLCM, LBP, first-order, shape) and clinical data. It is trained on the validation split and gives Low/Moderate/High risk. An ablation table compares CNN-only, radiomics-only, clinical-only and fusion. | `omnidx/radiomics.py`, `omnidx/fusion.py` |
| **Longitudinal monitoring** | Previous-vs-current comparison using ECC affine registration, probability change, ROI-area change and an aligned difference map. Status is one of Stable, Improvement indicator, Change detected or Requires clinical review. OmniCare trend rules cover lab values. | `omnidx/monitoring.py` |
| **Backend + database** | FastAPI with 30 endpoints and 12 SQLAlchemy tables. Role-based access (patient, clinician, researcher, admin), consent, audit log and a PDF screening report. Covers My Health Journey, Health Vault, OmniCare, follow-ups, alerts and dashboards. | `backend/` |
| **Tests** | 31 automated tests, covering the full flow from training to the API on a synthetic demo set. | `tests/` |

## Architecture

```
Medical image ─► validation ─► modality pipeline ─► segmentation (ROI)
                                     │                    │
                                     ▼                    ▼
                              CNN (ResNet/DenseNet/   radiomics features
                              EfficientNet) ─► Grad-CAM (+ ROI audit)
                                     │                    │
                     CNN embedding + probabilities ───────┤
                                                          ▼
                              clinical data ─► Random Forest ─► risk (Low/Moderate/High)
                                                          │
FastAPI ─► Screening record ─► PDF report ─► Health Vault ─► Clinician review (approve/override)
                                                          │
             My Health Journey ◄── diagnosis / treatment / follow-up / maintenance events
                                                          │
          Change & Recurrence Monitoring (scan vs scan)   +   OmniCare (lab trends) ─► alerts
```

Disease modules are plug-ins defined in `configs/modules.yaml`. To add a disease you add a block, put the images in `datasets/<key>/` and train. No API or UI code changes are needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Five-minute demo on synthetic data

This checks that everything works before you download the real datasets. The images are drawn shapes, not medical data, and every result is labelled DEMO.

```bash
python scripts/make_demo_data.py --out datasets/demo_xray
python -m omnidx.train --config configs/demo.yaml --module demo_xray --data datasets/demo_xray \
       --epochs 12 --no-pretrained --freeze-epochs 0 --lr 1e-3 --fusion
python -m pytest -q                  # 31 tests, about 5 minutes on CPU
```

## 2. Train the real disease modules

**Easiest route: Google Colab.** Open `notebooks/OmniDx_Colab_Training.ipynb` in Colab with a T4 GPU and run it top to bottom. It downloads the pneumonia, brain-tumor and HAM10000 skin datasets from Kaggle and prepares them with `scripts/prepare_kaggle.py`, using patient or lesion IDs for grouping. It then trains each module with the Random Forest fusion layer, and saves models, a results table and Grad-CAM figures to Google Drive.

The rest of this section is the manual route.

**Prepare the data.**

1. Download each Kaggle dataset and check that its licence allows academic use.
2. Fill in the `dataset:` fields in `configs/modules.yaml`.
3. Arrange the images in either of these layouts:

```
datasets/pneumonia/train/NORMAL/*.jpeg      datasets/skin_cancer/benign/*.jpg
datasets/pneumonia/train/PNEUMONIA/*.jpeg   datasets/skin_cancer/malignant/*.jpg
datasets/pneumonia/test/...                 (OmniDx makes a 70/15/15 split)
```

**Rename class folders.** Folder names become the class names. Rename them to match `positive_class` in the registry, for example `pneumonia` and `normal`.

**Group images by patient.** Add `datasets/<module>/metadata.csv` with the columns `filename,patient_id` and, optionally, clinical columns such as `age`, `spo2` or `bilirubin`. With this file, OmniDx splits by patient, and the Random Forest uses the clinical columns.

**Train.**

```bash
python -m omnidx.train --module pneumonia --data datasets/pneumonia --epochs 15 --fusion
python -m omnidx.train --module brain_tumor --data datasets/brain_tumor --epochs 20 --fusion
```

**Fix leakage errors.** Training stops if the leakage check fails. If the same patient appears in two splits, delete the dataset's own split folders and let OmniDx re-split by patient. If duplicate images cross splits, add `--drop-duplicates`.

**Use a GPU.** Training uses the GPU automatically when one is available. Google Colab works well: upload this folder and run the same commands.

**Outputs.** Each run writes these files to `models/<module>/<version>/`:

| File | Content |
|---|---|
| `metrics.json` | Accuracy and macro-F1 with 95% CI, per-class sensitivity/specificity, ROC-AUC, ECE, XAI audit |
| `confusion_matrix.png`, `roc_curve.png` | Plots for your report |
| `fusion_metrics.json`, `fusion_importance.png` | Random Forest ablation table and feature importance |
| `leakage.json`, `splits.csv`, `history.json` | Proof that the evaluation is clean and the training curve |

The newest run becomes the deployed version. An admin can switch versions through `POST /admin/modules/{key}/activate`.

Only report numbers that come from these files.

## 3. Run the API

```bash
uvicorn backend.main:app --reload
```

On the first run the server prints an admin password. Open http://127.0.0.1:8000/docs for the interactive API.

| Flow | Endpoints |
|---|---|
| Sign in, users | `POST /auth/login`, `POST /admin/users` |
| Patients, consent | `POST /patients`, `POST /patients/{id}/consent` |
| AI screening | `POST /patients/{id}/screenings` (image + module + clinical JSON), `GET /screenings/{id}/images/{original\|processed\|segmentation\|heatmap}` |
| Clinical review | `POST /screenings/{id}/review` (approve, or override with a required note) |
| Change monitoring | `GET /patients/{id}/compare?previous=&current=` |
| My Health Journey | `GET /patients/{id}/journey`, `POST /patients/{id}/events` |
| Health Vault | `POST/GET /patients/{id}/documents`, `GET /documents/{id}/download`, `DELETE /documents/{id}` |
| OmniCare | `POST /patients/{id}/metrics`, `GET /patients/{id}/omnicare`, `GET /patients/{id}/alerts` |
| Follow-up | `POST /patients/{id}/followups`, `PATCH /followups/{id}` |
| Dashboards | `GET /clinician/dashboard`, `GET /patients/{id}/dashboard`, `GET /omniwell` |
| Models, research | `GET /modules`, `GET /research/models`, `GET /admin/audit` |

**Database tables:** users, tokens, patients, medical_images, screenings (predictions), clinical_reviews, timeline_events, documents, metrics, follow_ups, alerts and audit_logs. Datasets, models and model versions live in the registry and the `models/` run folders.

SQLite is the default. Set `OMNIDX_DB=postgresql://...` for PostgreSQL.

## Safety rules built into the code

- **No fake predictions.** A module without a trained model returns `model_not_deployed`, not an invented result.
- **Screening, not diagnosis.** Results are labelled "AI-generated screening result" and stay `awaiting_clinician_review` until a clinician approves or overrides them. Every override needs a written reason.
- **Careful wording on change.** Change monitoring never says "relapse confirmed". It flags change for clinical review.
- **Consent before screening.** A patient must give consent before any AI screening runs.
- **Least access.** Patients see only their own records, and researchers see no patient data.
- **Protected files.** Files are stored outside public folders under random names and are only served to authorised users.
- **Audit log.** Every write is recorded, including model activation.
- **Correct dataset labels.** The COPD module states its dataset label: emphysema is not every form of COPD.

## Folder structure

```
omnidx/        CV + ML engine (preprocessing, segmentation, radiomics, data, models, train,
               evaluate, gradcam, fusion, monitoring, inference, report)
backend/       FastAPI app, database schema, security
configs/       disease-module registry (+ demo registry)
scripts/       synthetic demo data generator, Kaggle dataset preparation
notebooks/     Google Colab training notebook
tests/         31 pytest tests
```
