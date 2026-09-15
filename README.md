# Cardiovascular Disease Prediction

Machine learning project predicting the presence of cardiovascular disease from patient clinical & lifestyle data.

## Dataset
Cardiovascular Disease dataset — 70,000 patient records, 12 features (age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, physical activity) + binary target (cardio).

## Pipeline
1. Data pre-processing — dropped ID column, converted age from days to years, removed physiologically impossible blood-pressure rows, trimmed outliers (1st/99th percentile) in BP/height/weight, engineered a BMI feature.
2. EDA — target balance, age/BP/cholesterol/BMI vs disease, lifestyle factor plots.
3. Correlation matrix — identified blood pressure, age, and cholesterol as the strongest linear predictors.
4. Model comparison — Logistic Regression, KNN, Decision Tree, SVM, Random Forest trained on the same scaled train/test split (SVM trained on a subsample due to RBF kernel's poor scaling on 50k+ rows).
5. Final model — selected by accuracy & ROC-AUC, evaluated with a confusion matrix, classification report, ROC curve, and feature importances.

## How to run
pip install -r requirements.txt
jupyter notebook Cardiovascular_Disease_Prediction.ipynb

## Key results
Random Forest was the best-performing model (Accuracy ~73%, ROC-AUC ~0.80). Blood pressure, age, and cholesterol were the strongest predictors.

## Limitations & next steps
- Lifestyle features (smoking/alcohol/activity) are self-reported and noisy, weak correlation with the target despite being clinically relevant.
- No family history, ECG, or lab biomarker data included.
- Next: GridSearchCV hyperparameter tuning, gradient boosting models (XGBoost/LightGBM), SHAP explainability.
