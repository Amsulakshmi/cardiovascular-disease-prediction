# Cardiovascular Disease Prediction

Machine learning project predicting the presence of cardiovascular disease from patient clinical & lifestyle data.

## Dataset
[Cardiovascular Disease dataset (Kaggle)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) — 70,000 patient records, 11 features (age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, physical activity) + binary target (cardio).

## Pipeline
1. **Data cleaning** — dropped ID column, dropped duplicate rows, converted age from days to years, removed physiologically impossible blood-pressure rows.
2. **Train-test split first** — outlier trimming (1st/99th percentile) and BMI engineering are fit on the training set only, so the test set stays untouched and leakage-free.
3. **EDA & correlation** — computed on outlier-trimmed data (raw-data correlation is distorted by extreme `ap_hi` outliers).
4. **Model comparison** — Logistic Regression, KNN, Decision Tree, SVM, Random Forest compared via 5-fold cross-validation on an equal-sized sample, reporting mean ± std accuracy (not a single lucky split).
5. **Threshold tuning** — default 0.5 threshold missed a third of true disease cases (recall 0.66); tuned to 0.382 for ~0.80 recall, the right trade-off for a screening use case.
6. **Feature importance** — permutation importance (unbiased, model-agnostic) instead of default `feature_importances_`, computed on the held-out test set.

## How to run
pip install -r requirements.txt
jupyter notebook Cardiovascular_Disease_Prediction.ipynb

## Key results
SVM and Random Forest were statistically tied (0.7375 vs 0.7371 accuracy, within 1 std). Random Forest was chosen as the final model for practicality — it trains in seconds vs. SVM's poor scaling on 40k+ rows. `ap_hi` (systolic BP) was the dominant predictor in both the (outlier-trimmed) correlation and permutation importance, followed by cholesterol and age.

## Limitations & next steps
- Lifestyle features (smoking/alcohol/activity) are self-reported and noisy — weak correlation with the target despite being clinically relevant.
- No family history, ECG, or lab biomarker data included.
- BMI replaces raw height/weight to avoid redundant features.
- Next: GridSearchCV tuning inside the CV pipeline, gradient boosting models (XGBoost/LightGBM), SHAP explainability.
