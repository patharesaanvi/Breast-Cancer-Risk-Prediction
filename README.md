# Breast-Cancer-Risk-PredictionProject Overview

This project uses machine learning to predict breast cancer risk based on polygenic risk score (PRS) features. The goal was to develop and evaluate a classification model using genomic risk data.

Dataset

The dataset was provided through a Kaggle breast cancer risk prediction competition.

Training samples: 18,702

Test samples: 10,520

Features: 5,460 PRS Z-score features and predicted ancestry

Target variable: Breast cancer status

Methods

Used Python and scikit-learn to build a logistic regression classification model.

Applied feature standardization using StandardScaler.

Used one-hot encoding for predicted ancestry.

Evaluated regularization strengths to improve model performance.

Performed cross-validation to evaluate model performance.

Used area under the ROC curve (ROC-AUC) as the evaluation metric.

Results

Best mean 3-fold cross-validation AUC: 0.73431

Selected regularization parameter in that experiment: C = 0.0003

Competition baseline AUC: 0.70965

These results describe local model evaluation rather than a verified final Kaggle leaderboard score.

Tools and Libraries

Python

pandas

NumPy

scikit-learn

Logistic Regression

StandardScaler

OneHotEncoder

ROC-AUC

Project Files

The Python script contains the model training, preprocessing, evaluation, and prediction workflow.

The original competition data is not included in this repository.

Future Improvements

Compare logistic regression against additional machine learning models.

Explore feature selection and dimensionality reduction.

Evaluate model generalization across different ancestry groups.
