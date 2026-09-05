# MelanoScan HMS

**Development of a Hybrid Data-Centric Preprocessing Framework to Reduce Bias in Melanoma Detection**

Final Year Project — BSc (Hons) Computer Science with a specialism in Data Analytics
Asia Pacific University of Technology & Innovation (APU), 2026

---

## Overview

Melanoma is the deadliest form of skin cancer, but early detection raises the 5-year survival rate above 95%. AI-assisted detection is increasingly common — yet public dermatology datasets do not represent all patients equally. In HAM10000, paediatric patients account for roughly 3% of images. Models trained on this imbalance perform worst for the groups they saw least, producing less reliable diagnoses for exactly the patients who can least afford one.

Most published bias-mitigation work tackles a single demographic attribute at a time, usually skin tone. This project addresses **age × sex × lesion location together**, and deploys the resulting model inside a working clinical web application.

**Headline result:** melanoma sensitivity improved from **39.16% → 53.01%**, verified on an entirely unseen external dataset (ISIC 2020: **13.5% → 37.7%**, recovering 150 melanoma cases the baseline model missed).

---

## The Framework

Four components, each targeting a different mechanism of bias:

| # | Component | What it does |
|---|-----------|--------------|
| 1 | **Intersectional Stratified Sampling** | Splits train/val/test on combined age × sex × lesion-site keys rather than diagnosis alone, so rare subgroups appear in every split |
| 2 | **Adaptive Distribution-Aware Reweighting** | Scales per-sample training loss inversely to subgroup frequency — up to 44× for the rarest subgroup |
| 3 | **Quality-Controlled cGAN Augmentation** | A WGAN-GP conditioned on all four attributes generates synthetic images for 108 rare subgroups; discriminator-based ranking retains the top 50% (3,240 images kept) |
| 4 | **Melanoma-Priority Weighting (MelBoost)** | A 3× loss multiplier on the melanoma class, added after evaluation revealed a levelling-down effect |

### The levelling-down finding

Components 1–3 combined produced the **lowest fairness gap (EOD) of any model trained** — an apparent success. But melanoma sensitivity had fallen in *every* age group relative to baseline. The fairness metric improved because the model became equally worse for everyone, not better for the disadvantaged.

This is **levelling down**, and it is the central finding of the project: a fairness score read in isolation can be actively misleading. Component 4 was added specifically to correct it, lifting sensitivity across every age and sex subgroup rather than equalising them downward.

---

## Results

Six models trained on an identical EfficientNet-B0 architecture — only the preprocessing differs, isolating each component's contribution.

| Model | Accuracy | AUC | Melanoma Sens. | EOD (age) | EOD (sex) |
|-------|----------|-----|----------------|-----------|-----------|
| A — Standard Baseline | 0.8609 | 0.9818 | 0.3916 | 0.2237 | 0.1541 |
| B — Sampling Only | 0.7890 | 0.9492 | 0.2651 | 0.0419 | 0.0499 |
| C — Reweighting Only | 0.6533 | 0.8899 | 0.2771 | 0.0670 | 0.0830 |
| D — cGAN Only | 0.8053 | 0.9536 | 0.2831 | 0.0621 | 0.0285 |
| E — Full Framework | 0.7490 | 0.9346 | 0.3012 | 0.0114 | 0.0070 |
| **E — MelBoost 3.0 (deployed)** | **0.7327** | **0.9324** | **0.5301** | **0.1154** | **0.0273** |

**Reading these numbers:** Model A has the highest accuracy but misses roughly 61 of every 100 melanomas — HAM10000 is ~67% benign nevi, so accuracy rewards getting the majority class right. Melanoma sensitivity is the clinically meaningful metric here, and it is what the deployed model optimises for.

### External validation — ISIC 2020

Evaluated on 3,584 held-out images (584 melanoma + 3,000 benign, sampled from the deduplicated 33,126-image ISIC 2020 set, with zero overlap against HAM10000):

| Metric | Model A | Model E (MelBoost 3.0) |
|--------|---------|------------------------|
| Melanoma sensitivity | 13.5% | **37.7%** |
| Melanomas missed per 100 | 86 | **62** |

Genuine uplift confirmed on all three protected axes — the worst-performing subgroup improved in absolute terms, not merely relative to the best.

---

## MelanoScan HMS

The deployed model runs inside a full-stack hospital management system.

**Stack:** Flask (Python) · React 18 · SQLAlchemy · SQLite

- **Doctor Portal** — Detection Analysis, Patient Management, Appointments, Messages, Model Performance, Analytics & Fairness Dashboard, Notifications
- **Patient Portal** — Scan Analysis with downloadable PDF report, My Results, My Appointments, Medical History, Messages
- **Shared prediction engine** — every prediction, from either portal, routes through the same validated model
- **Fairness dashboard** — OLAP-style drill-down showing model performance across demographic subgroups, so clinicians can see *where* the model is weaker, not just how confident it is

---

## Repository Structure

> **TODO:** adjust to match your actual folder layout.

```
├── notebooks/
│   ├── 1st_Half_Notebook_Source_Code.ipynb    # Data import, preprocessing, hybrid framework (Components 1-3)
│   ├── 2nd_Half_Notebook_Source_Code.ipynb    # Model building, training, evaluation (6-model ablation)
│   └── phase4_isic2020_validation.ipynb       # External validation on ISIC 2020
├── melanoscan_hms/                            # Flask + React application
│   ├── app.py
│   ├── models.py                              # 8 SQLAlchemy models
│   ├── routes/
│   │   └── api.py
│   ├── ml/
│   │   └── predict.py                         # Shared prediction engine
│   └── static/                                # React frontend
├── phase1_outputs/
│   ├── models/                                # Trained .h5 model files
│   ├── splits/                                # Train/val/test CSVs
│   ├── results/                               # Metrics, charts, all_results.json
│   ├── cgan/                                  # Generator checkpoints, training curves
│   └── synthetic_images/                      # cGAN output per rare subgroup
├── docs/
│   └── Setup_and_Run_Guide.pdf
├── requirements.txt
└── README.md
```

---

## Getting Started

> **TODO:** verify these commands against your actual setup guide.

### Running MelanoScan HMS

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

> **Important:** MelanoScan is a single Flask application. Keep every file and folder together — the backend and frontend are not separate projects and must not be split into separate directories.

### Reproducing the research

The notebooks are also published on Kaggle with full outputs:

- [1st Half — Preprocessing & Hybrid Framework](https://www.kaggle.com/code/bobbi12235/notebook-split-1st-half)
- [2nd Half — Model Building & Evaluation](https://www.kaggle.com/code/bobbi12235/notebook-split-2nd-half-fixed)

A GPU accelerator is required — the cGAN and CNN training cells will not run in reasonable time on CPU. The second notebook expects the first notebook's outputs (splits, sample weights, synthetic images) as an input dataset.

---

## Datasets

| Dataset | Use | Size |
|---------|-----|------|
| [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | Training and internal evaluation | 10,015 dermoscopic images, 7 classes |
| [ISIC 2020](https://challenge2020.isic-archive.com/) | External validation | 3,584 evaluated (deduplicated against HAM10000) |

---

## Limitations

Stated plainly, because they matter for how these results should be read:

- **Skin tone is not addressed.** HAM10000 lacks reliable Fitzpatrick labels, so this axis could not be measured, let alone mitigated. It remains an open blind spot.
- **Dataset scale.** 10,015 images total; the rarest intersectional subgroups still have very few genuine (non-synthetic) examples.
- **Fairness metric scope.** Evaluation centres on TPR parity and EOD. Full Equalized Odds (the false-positive-rate side) was not implemented.
- **No clinical usability testing.** The HMS was tested with fictional patient and doctor records, not practising dermatologists or real patient data.
- **EOD by lesion location widened** for the deployed model, even though every location subgroup improved in absolute terms — the best-performing group improved faster than the worst. Location is the axis where this framework has the least clean result.

---

## Future Work

- Incorporate ISIC Challenge and Fitzpatrick17k / DDI datasets to add real images and cover skin tone
- Add FPR-based fairness (Equalized Odds) alongside EOD; test architectures beyond EfficientNet-B0
- Run a structured usability study with practising dermatologists on real patient data
- Incorporate structured clinical metadata (patient history, ABCDE criteria) for multi-modal prediction
- Add explainability tooling (Grad-CAM, saliency mapping) so clinicians can verify model focus

---

## Acknowledgements

Supervisor: **Dr. Kulothunkan Palasundram**
Second Marker: **Ms. Hema Latha Krishna Nair**

Aligned with **UN SDG 3** (Good Health and Well-Being) and **UN SDG 10** (Reduced Inequalities).

---

## Author

**Ramaneiss Pillai** — TP070818
[LinkedIn](https://www.linkedin.com/in/ramaneiss)

---

## License

> **TODO:** add a license if you want one. MIT is the usual default for student projects.
> Note that HAM10000 and ISIC 2020 carry their own dataset licenses (CC BY-NC 4.0 for HAM10000) — your license applies to your code, not the data.
