
# Breast Cancer Risk Prediction Using Polygenic Risk Scores

## Overview

This project uses machine learning to predict breast cancer risk based on polygenic risk scores (PRS). It was completed as part of a Kaggle bioinformatics competition.

## Dataset

The dataset contains:
- 18,702 training samples
- 10,520 test samples
- 5,460 PRS features and predicted ancestry
- Target variable: `breast_cancer`

The original competition data is not included in this repository.

## Methods

- Preprocessed PRS features using `StandardScaler` and encoded predicted ancestry using `OneHotEncoder`.
- Trained a logistic regression model using Python and scikit-learn.
- Tested different regularization strengths (`C` values).
- Evaluated model performance using 3-fold cross-validation and ROC-AUC.
- Generated predicted probabilities for the test dataset.

## Results

| Model | Regularization (C) | Mean CV ROC-AUC |
|---|---|---|
| Logistic Regression | 0.0003 | 0.73431 |

The competition baseline AUC was 0.70965. The reported model result is from local cross-validation, not the final Kaggle leaderboard.

## Tools

Python, NumPy, Pandas, scikit-learn


## Future Improvements

- Compare logistic regression with other machine learning algorithms.
- Explore feature selection and dimensionality reduction.
- Investigate model performance across predicted ancestry groups.
- Improve model generalization through additional validation experiments.
