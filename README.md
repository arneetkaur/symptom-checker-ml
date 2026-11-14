# AI-Based Symptom-to-Disease Classification System for Preliminary Health Screening

## Background
Many people search their symptoms online before seeing a clinician. This
project builds a machine learning screening aid that takes reported
symptoms and returns the three most likely diseases with probabilities and
an explanation of which symptoms drove each prediction. It is a university
project (ECE 698C, CSUN) and is NOT a medical diagnosis tool.

## Introduction
- Data: Kaggle "Disease Prediction Using Machine Learning" dataset
  (132 binary symptoms, 41 diseases). The 4,920 rows contain only 304 unique
  cases, so duplicates are removed before the train/test split.
- Model: Logistic Regression with top-3 predictions, coefficient-based
  drivers and SHAP explanations; compared with Random Forest and linear SVM.
- Experiments: leakage check, 5-fold cross-validation, and a robustness test
  with incomplete symptom reports.

## Dataset
Download Training.csv from
https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
and place it in the project folder.

## Usage
pip install -r requirements.txt
python baseline.py              # leakage check and cross-validation
python pipeline.py              # trains and saves the model
python robustness_repeated.py   # robustness experiment
python plot_robustness.py       # robustness figure
streamlit run app.py            # demo app (if app.py is included)

## Report
The accompanying report is part of ECE 698C at California State University,
Northridge. A link to the report will be added here.