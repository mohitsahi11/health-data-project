# Heart Disease Risk Prediction — Real-World Data Project (Health Domain)

An end-to-end data science project on a real patient-records dataset: exploratory
data analysis (EDA), predictive modeling, and evaluation to identify which
patients are at risk of heart disease.

## Problem Statement

Given anonymized clinical measurements for 303 patients (age, blood pressure,
cholesterol, ECG results, etc.), predict whether a patient has heart disease
(`target = 1`) or not (`target = 0`). This is a binary classification problem.

## Dataset

- **Source:** [UCI Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Size:** 303 patient records, 13 features + 1 target column
- **File:** [`data/heart_disease.csv`](data/heart_disease.csv)

| Column | Description |
|---|---|
| `age` | Age of the patient in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting electrocardiographic results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0–4) colored by fluoroscopy |
| `thal` | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| `target` | 1 = heart disease present, 0 = no heart disease |

## Repository Structure

```
health-data-project/
├── data/
│   └── heart_disease.csv          # raw dataset
├── src/
│   ├── data_loader.py             # load + clean data
│   ├── eda.py                     # exploratory analysis & charts
│   ├── train_model.py             # model training, tuning, evaluation
│   └── main.py                    # runs the full pipeline end-to-end
├── visuals/                       # all generated charts (PNG)
├── results/
│   └── model_metrics.json         # final metrics for all models
├── report/
│   └── FINDINGS.md                # detailed write-up of findings & conclusions
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
cd src
python main.py
```

This runs the full pipeline: loads and cleans the data, generates all EDA
charts into `visuals/`, trains and tunes the models, and saves final metrics
to `results/model_metrics.json`.

## Methodology

1. **Data Cleaning** — checked for missing values (none found) and duplicate
   rows (1 duplicate removed), leaving 302 clean records.
2. **Exploratory Data Analysis** — distribution of the target class, age vs.
   diagnosis, correlation between features, chest pain type vs. diagnosis,
   and age vs. maximum heart rate achieved.
3. **Modeling** — compared three baseline classifiers (Logistic Regression,
   Random Forest, K-Nearest Neighbors) on a stratified 80/20 train-test
   split with scaled features.
4. **Tuning** — used `GridSearchCV` (5-fold cross-validation) to tune the
   Random Forest's `n_estimators`, `max_depth`, and `min_samples_split`.
5. **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC,
   confusion matrix, and feature importance for the final model.

## Results Summary

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.787 | 0.763 | 0.879 | 0.817 | 0.865 |
| Random Forest | 0.803 | 0.756 | 0.939 | 0.838 | 0.882 |
| K-Nearest Neighbors | 0.820 | 0.790 | 0.909 | 0.845 | 0.891 |
| **Tuned Random Forest (final)** | **0.803** | **0.756** | **0.939** | **0.838** | **0.903** |

5-fold cross-validation accuracy of the tuned model: **0.834 (± 0.056)**

Full details, visualizations, and conclusions are in
[`report/FINDINGS.md`](report/FINDINGS.md).

## Tech Stack

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn

## Author

Mohit
