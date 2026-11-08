
# Breast Cancer Risk Prediction Using Polygenic Risk Scores

## Overview

This project focuses on predicting breast cancer risk using polygenic risk scores (PRS) and machine learning. The goal was to develop a classification model that could predict breast cancer status based on genomic risk features.

The project was completed as part of a bioinformatics machine learning competition on Kaggle.

## Dataset

The dataset was provided through the Breast Cancer Risk Prediction Kaggle competition.

- **Training samples:** 18,702
- **Test samples:** 10,520
- **PRS features:** 5,460
- **Additional feature:** Predicted ancestry
- **Target variable:** `breast_cancer`

The original competition datasets are not included in this repository.

## Methods

The model was developed using Python and scikit-learn.

### Data Preprocessing
- Standardized numerical PRS features using `StandardScaler`.
- Encoded the categorical ancestry feature using `OneHotEncoder`.
- Used a preprocessing pipeline to apply transformations consistently.

### Model Development
- Trained a logistic regression classifier.
- Tested different regularization strengths (`C` values).
- Used stratified cross-validation to evaluate model performance.
- Generated predicted probabilities for the test dataset.

### Model Evaluation
- Evaluated predictions using ROC-AUC (Area Under the Receiver Operating Characteristic Curve).
- Compared different regularization strengths to identify a suitable model configuration.

## Results

The best mean ROC-AUC from the 3-fold cross-validation experiments was:

| Model | Regularization (C) | Mean CV ROC-AUC |
|-------|--------------------|----------------|
| Logistic Regression | 0.0003 | 0.73431 |

The competition baseline AUC was 0.70965.

These results represent local cross-validation performance, not a final Kaggle leaderboard score.

## Tools and Libraries

- Python
- NumPy
- Pandas
- scikit-learn
- Logistic Regression
- StandardScaler
- OneHotEncoder
- ROC-AUC

## Project Structure

```text
Breast-Cancer-Risk-Prediction/
├── README.md
├── breast_cancer_model.py
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/patharesaanvi/Breast-Cancer-Risk-Prediction.git
cd Breast-Cancer-Risk-Prediction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Model

After obtaining the original competition data and placing `train.csv` and `test.csv` in the locations expected by the script, run:

```bash
python breast_cancer_model.py
```

The script trains the model and generates breast cancer risk predictions for the test dataset.

## Future Improvements

- Compare logistic regression with other machine learning algorithms.
- Explore feature selection and dimensionality reduction.
- Investigate model performance across predicted ancestry groups.
- Improve model generalization through additional validation experiments.
