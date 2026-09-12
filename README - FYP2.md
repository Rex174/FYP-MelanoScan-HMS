<div align="center">

# 🔬 MelanoScan HMS

### Development of a Hybrid Data-Centric Preprocessing Framework to Reduce Bias in Melanoma Detection

**Final Year Project** · BSc (Hons) Computer Science with a specialism in Data Analytics
Asia Pacific University of Technology & Innovation (APU) · 2026

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-WGAN--GP-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-D71F00?style=flat-square&logo=sqlalchemy&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-DB-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

<br>

![Architecture](https://img.shields.io/badge/Architecture-EfficientNet--B0-00938D?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000-D45B3D?style=flat-square)
![Validation](https://img.shields.io/badge/External%20Validation-ISIC%202020-D45B3D?style=flat-square)
![SDG](https://img.shields.io/badge/UN%20SDG-3%20%26%2010-4C9F38?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

</div>

---

## 📌 The Question Behind This Project

> **An AI model is 86% accurate — but misses 61 of every 100 melanoma cases.**
> **Is that a good model? Is it a fair one?**

Melanoma is the deadliest form of skin cancer, yet early detection pushes the 5-year survival rate **above 95%**. AI-assisted detection is becoming routine — but public dermatology datasets don't represent every patient equally. In HAM10000, paediatric patients account for roughly **3%** of all images.

When a model learns from imbalance, it quietly performs worst for the groups it saw least. Those patients receive the least reliable diagnosis — precisely the people who can least afford one.

Most published bias-mitigation work tackles **one attribute at a time**, usually skin tone. This project addresses **age × sex × lesion location together** — and deploys the result inside a working clinical system.

<div align="center">

### 🎯 Headline Result

| | Baseline | **Deployed Model** |
|:---|:---:|:---:|
| **Melanoma sensitivity** (internal) | 39.16% | **53.01%** 🔺 |
| **Melanoma sensitivity** (unseen ISIC 2020) | 13.5% | **37.7%** 🔺 |
| **Melanomas recovered** | — | **+150 cases** |

</div>

---

## 🧩 The Framework

Four components, each targeting a **different mechanism** of bias.

<table>
<tr>
<td width="60" align="center"><h3>01</h3></td>
<td><b>📊 Intersectional Stratified Sampling</b><br>
Splits train/val/test on combined <code>age × sex × lesion-site</code> keys rather than diagnosis alone — so rare subgroups appear in <i>every</i> split, and evaluation reflects the real demographic spread.</td>
</tr>
<tr>
<td width="60" align="center"><h3>02</h3></td>
<td><b>⚖️ Adaptive Distribution-Aware Reweighting</b><br>
Scales per-sample training loss <i>inversely</i> to subgroup frequency — up to <b>44×</b> for the rarest subgroup, forcing the model to learn from scarce groups instead of ignoring them.</td>
</tr>
<tr>
<td width="60" align="center"><h3>03</h3></td>
<td><b>🎨 Quality-Controlled cGAN Augmentation</b><br>
A <b>WGAN-GP</b> conditioned on all four attributes generates synthetic images for <b>108 rare subgroups</b>; discriminator-based ranking retains the top 50% — <b>3,240 images</b> kept.</td>
</tr>
<tr>
<td width="60" align="center"><h3>04</h3></td>
<td><b>🎯 Melanoma-Priority Weighting (MelBoost)</b><br>
A <b>3× loss multiplier</b> on the melanoma class — added <i>after</i> evaluation exposed a problem the other three components created together. 👇</td>
</tr>
</table>

---

## 💡 The Levelling-Down Discovery

> **The most valuable result was the one I didn't expect.**

Components 1–3 combined produced the **lowest fairness gap (EOD) of any model trained** — on paper, an unambiguous success.

Then I broke the results down by subgroup. Melanoma sensitivity had **fallen in every single age group** compared to baseline.

The fairness metric improved because the model became **equally worse for everyone** — not better for the disadvantaged.

<div align="center">

```
❌  LEVELLING DOWN          ✅  GENUINE UPLIFT
    gap closes because          gap closes because
    everyone gets worse         the worst-off improve
```

</div>

This is the central finding of the project: **a fairness score read in isolation can be actively misleading.** Component 4 exists specifically to correct it — lifting sensitivity across every age and sex subgroup rather than equalising them downward.

---

## 📈 Results

Six models · **identical EfficientNet-B0 architecture** · only the preprocessing differs, isolating each component's contribution.

| Model | Accuracy | AUC | Melanoma Sens. | EOD (age) | EOD (sex) |
|:---|:---:|:---:|:---:|:---:|:---:|
| A — Standard Baseline | `0.8609` | `0.9818` | `0.3916` | `0.2237` | `0.1541` |
| B — Sampling Only | `0.7890` | `0.9492` | `0.2651` | `0.0419` | `0.0499` |
| C — Reweighting Only | `0.6533` | `0.8899` | `0.2771` | `0.0670` | `0.0830` |
| D — cGAN Only | `0.8053` | `0.9536` | `0.2831` | `0.0621` | `0.0285` |
| E — Full Framework | `0.7490` | `0.9346` | `0.3012` | `0.0114` | `0.0070` |
| **🏆 E — MelBoost 3.0** | **`0.7327`** | **`0.9324`** | **`0.5301`** | **`0.1154`** | **`0.0273`** |

> **⚠️ Reading these numbers:** Model A has the highest accuracy but misses ~61 of every 100 melanomas. HAM10000 is ~67% benign nevi, so accuracy rewards getting the *majority* class right. **Melanoma sensitivity is the clinically meaningful metric** — and it's what the deployed model optimises for.

### 🧪 External Validation — ISIC 2020

Evaluated on **3,584 held-out images** (584 melanoma + 3,000 benign), sampled from the deduplicated 33,126-image ISIC 2020 set with **zero overlap** against HAM10000.

| Metric | Model A | **Model E (MelBoost 3.0)** |
|:---|:---:|:---:|
| Melanoma sensitivity | 13.5% | **37.7%** |
| Melanomas missed per 100 | 86 | **62** |

✅ **Genuine uplift confirmed on all three protected axes** — the worst-performing subgroup improved in *absolute* terms, not merely relative to the best.

---

## 🏥 MelanoScan HMS

The deployed model runs inside a full-stack hospital management system.

<table>
<tr>
<td width="50%" valign="top">

### 👨‍⚕️ Doctor Portal
- Detection Analysis
- Patient Management
- Appointments & Messages
- Model Performance
- **Analytics & Fairness Dashboard**
- Notifications

</td>
<td width="50%" valign="top">

### 🧑 Patient Portal
- Scan Analysis + PDF report
- My Results
- My Appointments
- Medical History
- Messages & Notifications

</td>
</tr>
</table>

**🔗 Shared prediction engine** — every prediction, from either portal, routes through the same validated model.

**📊 Fairness dashboard** — OLAP-style drill-down showing performance across demographic subgroups, so clinicians can see *where* the model is weaker, not just how confident it is.

---

## 📁 Repository Structure

> ⚙️ **TODO:** adjust to match your actual folder layout.

```
📦 MelanoScan-HMS
├── 📓 notebooks/
│   ├── 1st_Half_Notebook_Source_Code.ipynb   → Data import, preprocessing, framework (Components 1–3)
│   ├── 2nd_Half_Notebook_Source_Code.ipynb   → Model building, training, 6-model ablation
│   └── phase4_isic2020_validation.ipynb      → External validation on ISIC 2020
├── 🏥 melanoscan_hms/
│   ├── app.py
│   ├── models.py                             → 8 SQLAlchemy models
│   ├── routes/api.py
│   ├── ml/predict.py                         → Shared prediction engine
│   └── static/                               → React frontend
├── 📊 phase1_outputs/
│   ├── models/                               → Trained .h5 files
│   ├── splits/                               → Train/val/test CSVs
│   ├── results/                              → Metrics, charts, all_results.json
│   ├── cgan/                                 → Generator checkpoints, curves
│   └── synthetic_images/                     → cGAN output per rare subgroup
├── 📄 docs/
│   └── Setup_and_Run_Guide.pdf
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

> ⚙️ **TODO:** verify these commands against your actual setup guide.

```bash
# 1️⃣  Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2️⃣  Install dependencies
pip install -r requirements.txt

# 3️⃣  Launch
python app.py
```

🌐 Open **http://127.0.0.1:5000**

> **⚠️ Important:** MelanoScan is a *single* Flask application. Keep every file and folder together — the backend and frontend are not separate projects and must not be split into separate directories.

### 🔬 Reproducing the Research

Both notebooks are published on Kaggle with full outputs:

[![1st Half](https://img.shields.io/badge/Kaggle-1st%20Half%20·%20Preprocessing%20%26%20Framework-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/bobbi12235/notebook-split-1st-half)

[![2nd Half](https://img.shields.io/badge/Kaggle-2nd%20Half%20·%20Model%20Building%20%26%20Evaluation-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/bobbi12235/notebook-split-2nd-half-fixed)

⚡ A **GPU accelerator is required** — cGAN and CNN training will not complete in reasonable time on CPU. The second notebook expects the first notebook's outputs (splits, sample weights, synthetic images) as an input dataset.

---

## 🗂️ Datasets

| Dataset | Use | Size |
|:---|:---|:---|
| [**HAM10000**](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | Training & internal evaluation | 10,015 dermoscopic images · 7 classes |
| [**ISIC 2020**](https://challenge2020.isic-archive.com/) | External validation | 3,584 evaluated · deduplicated vs HAM10000 |

---

## ⚠️ Limitations

Stated plainly, because they matter for how these results should be read.

| | Limitation |
|:---:|:---|
| 🎨 | **Skin tone is not addressed.** HAM10000 lacks reliable Fitzpatrick labels, so this axis could not be measured — let alone mitigated. An open blind spot. |
| 📉 | **Dataset scale.** 10,015 images total; the rarest intersectional subgroups still have very few genuine (non-synthetic) examples. |
| 📏 | **Fairness metric scope.** Evaluation centres on TPR parity and EOD. Full Equalized Odds (the FPR side) was not implemented. |
| 🏥 | **No clinical usability testing.** The HMS was tested with fictional patient and doctor records — not practising dermatologists or real patient data. |
| 📍 | **EOD by lesion location widened** for the deployed model, even though every location subgroup improved in absolute terms — the best group improved faster than the worst. Location is the axis with the least clean result. |

---

## 🔮 Future Work

- 🗃️ Incorporate **ISIC Challenge** and **Fitzpatrick17k / DDI** datasets to add real images and cover skin tone
- 📐 Add **FPR-based fairness (Equalized Odds)** alongside EOD; test architectures beyond EfficientNet-B0
- 🩺 Run a structured **usability study** with practising dermatologists on real patient data
- 📋 Incorporate structured **clinical metadata** (patient history, ABCDE criteria) for multi-modal prediction
- 🔍 Add **explainability tooling** (Grad-CAM, saliency mapping) so clinicians can verify model focus

---

## 🙏 Acknowledgements

**Supervisor** · Dr. Kulothunkan Palasundram
**Second Marker** · Ms. Hema Latha Krishna Nair

<div align="center">

![SDG 3](https://img.shields.io/badge/SDG%203-Good%20Health%20%26%20Well--Being-4C9F38?style=for-the-badge)
![SDG 10](https://img.shields.io/badge/SDG%2010-Reduced%20Inequalities-DD1367?style=for-the-badge)

</div>

---

<div align="center">

### 👤 Author

**Ramaneiss Pillai** · TP070818

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ramaneiss)

<br>

> ⚙️ **TODO:** add a license if you want one — MIT is the usual default for student projects.
> Note that HAM10000 and ISIC 2020 carry their own dataset licenses (HAM10000 is CC BY-NC 4.0). Your license covers your code, not the data.

</div>
