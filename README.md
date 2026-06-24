# Utilizing Machine Learning for Pediatric Sepsis and Septic Shock Diagnosis in Resource-Limited Settings
 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
 
**Manuscript:** Bunch K, Shaima SN, Mamun GMS, Jarabana SG, Gainey M, Rahman ASMH, Genisca A, Jindal A, Kadakia N, Sarmin M, Afroze F, Levine AC, Chisti MJ, Garbern SC. *Utilizing Machine Learning for Pediatric Sepsis and Septic Shock Diagnosis in Resource-Limited Settings.* Pediatric Reports. 2026.
 
**Journal:** Pediatric Reports (MDPI) | **Manuscript ID:** pediatrrep-4358533
 
---
 
## Overview
 
This repository contains the complete analysis pipeline for a retrospective secondary analysis evaluating machine learning-based **concurrent diagnostic classifiers** for pediatric sepsis and septic shock in a resource-limited inpatient setting. Models were developed to serve as a simplified operational approximation of the 2024 Phoenix Sepsis Score (PSS) using readily available clinical and laboratory variables.
 
**Study site:** Intensive Care Unit, International Centre for Diarrhoeal Disease Research, Bangladesh (icddr,b), Dhaka, Bangladesh  
**Cohort:** N = 100 children aged 2 months–17 years with clinician-suspected infection meeting ≥2 SIRS criteria (February–December 2022)  
**Outcomes:** Sepsis (PSS ≥ 2) and Septic Shock (PSS ≥ 2 with cardiovascular sub-score ≥ 1)
 
> **Important framing note:** All predictors and outcome labels were derived from the same 24-hour assessment window. This is a concurrent diagnostic classification study, not a prospective prediction model. Results should be considered hypothesis-generating and require external validation before any clinical use.
 
---
 
## Repository Structure
 
```
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── notebooks/
│   └── Pediatric_Sepsis_ML.ipynb          # Main analysis notebook (Google Colab)
│
├── src/
│   ├── 01_dataset_construction.py      # Steps 1–8: CRF merging, imputation, PSS scoring
│   ├── 02_demographics.py              # Table 1 and hospital outcomes
│   ├── 03_ml_analysis.py               # Core ML pipeline (one file, change CONFIG block)
│   └── 04_combine_csv_outputs.py       # Aggregates per-analysis CSVs into manuscript tables
│
├── config/
│   └── feature_sets.py                 # CLINLAB_FEATURES and NONLAB_FEATURES definitions
│
└── outputs/
    └── csv/                            # Per-analysis CSV outputs (not tracked — see Data section)
```
 
---
 
## Methods Summary
 
### Dataset Construction
Raw REDCap data (3,992 rows × 999 columns, 100 patients) is processed through an 8-step pipeline:
 
| Step | Description |
|------|-------------|
| 1 | Admission CRF3 extraction (vitals, GCS, demographics) |
| 2 | Periodic vitals (CRF10/CRF15 merge) |
| 3 | Treatments (CRF4 admission + CRF11 periodic) |
| 4 | Laboratory values (CRF5 admission + CRF12 periodic) |
| 5 | Organ scoring (CRF7 admission + CRF14 periodic) |
| 6 | Union-grid merge into 4-hour time slots (all sources) |
| 7 | Imputation: pre-admission lab attribution, linear interpolation (48h gap cap), LOCF/NOCB |
| 8 | 77-sentinel replacement, forward/backward fill, worst-value aggregation over 24-hour window |
 
### Phoenix Sepsis Score (PSS) Computation
PSS components computed per the 2024 consensus criteria [Schlapbach et al., 2024]:
- **Respiratory** (0–3): SF-ratio with ventilation status
- **Cardiovascular** (0–6): vasopressors + lactate
- **MAP-for-age** (0–2): age-stratified MAP thresholds  
- **Coagulation** (0–2): platelets + INR
- **Neurological** (0–2): GCS + fixed pupils (PELOD pupillary score = 5)
D-dimer and fibrinogen were unavailable and scored as 0 per PSS guidelines. Sepsis = PSS ≥ 2; Septic shock = sepsis + cardiovascular sub-score ≥ 1.
 
### Feature Sets
 
| Feature Set | Variables (n) |
|-------------|---------------|
| **Clinical + Laboratory** | SpO₂:FiO₂ ratio, WBC, temperature, platelets, heart rate, hemoglobin, respiratory rate, systolic BP, age (months), GCS |
| **Non-Laboratory (Clinical)** | SpO₂:FiO₂ ratio, temperature, heart rate, respiratory rate, systolic BP, age (months), GCS |
 
### Machine Learning Pipeline
- **Models:** Decision Tree, Random Forest, SVM (linear), Kernel SVM (RBF), Naïve Bayes, KNN, Logistic Regression
- **Hyperparameter optimization:** Optuna (50 trials, AUPRC scoring, stratified group k-fold)
- **Cross-validation:** Stratified group 5-fold (N_REPEATS = 40), patient-clustered
- **Confidence intervals:** 2,000-resample patient-level clustered bootstrap on fixed out-of-fold predictions
- **Optimism correction:** Harrell/Steyerberg bootstrap (apparent − mean[boot_train − boot_test])
- **Thresholds:** Three pre-specified operating points: 0.50 (reference), inner-CV F1-optimal (train only), 0.30 (screening)
- **Class imbalance:** `class_weight='balanced'` throughout; SVMs wrapped in `CalibratedClassifierCV(method='sigmoid', cv=3)`
- **Scaling:** `StandardScaler` applied strictly within each training fold
---
 
## Results Summary
 
### Headline Model: Logistic Regression (AUROC/AUPRC with 95% bootstrap CI)
 
| Feature Set | Outcome | AUROC | AUPRC | Brier |
|-------------|---------|-------|-------|-------|
| Clinical+Lab | Sepsis | 0.945 (0.881–0.986) | 0.941 (0.880–0.981) | 0.086 (0.049–0.132) |
| Non-Lab | Sepsis | 0.945 (0.890–0.983) | 0.942 (0.884–0.979) | 0.163 (0.149–0.178) |
| Clinical+Lab | Septic Shock | 0.870 (0.761–0.952) | 0.793 (0.621–0.908) | 0.141 (0.119–0.166) |
| Non-Lab | Septic Shock | 0.878 (0.758–0.967) | 0.822 (0.664–0.929) | 0.119 (0.080–0.171) |
 
> **Caution:** Septic shock results must be interpreted cautiously given the small number of positive cases (N=23) and predictor-outcome overlap with PSS components.
 
---
 
## Installation
 
### Requirements
```bash
pip install -r requirements.txt
```
 
### Running in Google Colab (recommended)
The analysis was developed and validated in Google Colab Pro. To replicate:
 
1. Upload `Dataset_Analysis.ipynb` to Google Colab
2. Mount Google Drive and update `FILE_PATH` in the CONFIG block to point to your copy of the dataset
3. Run cells sequentially from top to bottom
4. Outputs are saved to `MyDrive/Pediatric Sepsis ML Model/csv/`
### Running locally
```bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/pediatric-sepsis-ml
cd pediatric-sepsis-ml
pip install -r requirements.txt
# Update FILE_PATH in src/01_dataset_construction.py before running
python src/01_dataset_construction.py
python src/03_ml_analysis.py
```
 
---
 
## Data Availability
 
The raw dataset (`REMEDIES_R21_Final_Dataset.csv`) is **not publicly available** due to patient privacy and IRB restrictions. Data were collected under protocols approved by the icddr,b Institutional Review Board (Research Review Committee and Ethical Review Committee) and the Rhode Island Hospital IRB.
 
Researchers interested in accessing the data for collaborative purposes should contact the corresponding author (stephanie_garbern@brown.edu).
 
---
 
## Reproducing the Analysis
 
Each of the four analyses (Clinical+Lab Sepsis, Non-Lab Sepsis, Clinical+Lab Septic Shock, Non-Lab Septic Shock) uses the same code block. Change only the `CONFIG` block at the top of `src/03_ml_analysis.py`:
 
```python
# Clinical+Lab Sepsis (default)
FEATURE_SET_NAME = "Clinical+Lab"
TARGET           = "Sepsis"
FEATURES         = CLINLAB_FEATURES
 
# Non-Lab Sepsis
FEATURE_SET_NAME = "Clinical"
TARGET           = "Sepsis"
FEATURES         = NONLAB_FEATURES
 
# Clinical+Lab Septic Shock
FEATURE_SET_NAME = "Clinical+Lab"
TARGET           = "Septic_shock"
FEATURES         = CLINLAB_FEATURES
 
# Non-Lab Septic Shock
FEATURE_SET_NAME = "Clinical"
TARGET           = "Septic_shock"
FEATURES         = NONLAB_FEATURES
```
 
Set `N_BOOT = 200` for rapid testing; use `N_BOOT = 2000` for final analysis (runtime ~20–40 min per analysis on Colab Pro).
 
---
 
## Citation
 
If you use this code or build on this work, please cite:
 
```bibtex
@article{bunch2026pediatric,
  title   = {Utilizing Machine Learning for Pediatric Sepsis and Septic Shock 
             Diagnosis in Resource-Limited Settings},
  author  = {Bunch, Kaden and Shaima, Shamsun Nahar and Mamun, Gazi Md. Salahuddin 
             and Jarabana, Sai Gopal and Gainey, Monique and Rahman, Abu Sayem Mirza 
             Md. Hasibur and Genisca, Alicia and Jindal, Atin and Kadakia, Nidhi and 
             Sarmin, Monira and Afroze, Farzana and Levine, Adam C and Chisti, 
             Mohammod Jobayer and Garbern, Stephanie Chow},
  journal = {Pediatric Reports},
  year    = {2026},
  publisher = {MDPI},
  note    = {Manuscript ID: pediatrrep-4358533}
}
```
 
See also `CITATION.cff` for machine-readable citation metadata.
 
---
 
## License
 
This code is released under the [MIT License](LICENSE). The underlying dataset is not included and remains subject to IRB restrictions (see Data Availability above).
 
---
 
## Contact
 
**Corresponding author:** Kaden Bunch  
Warren Alpert Medical School of Brown University  
kaden_bunch@brown.edu
 
**Principal investigator:** Stephanie Chow Garbern, MD, MPH  
Associate Professor of Emergency Medicine, Brown University  
stephanie_garbern@brown.edu
