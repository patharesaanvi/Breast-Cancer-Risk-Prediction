
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Read column names without loading the data
print("Reading column names...")

train_columns = pd.read_csv("train.csv", nrows=0).columns
test_columns = pd.read_csv("test.csv", nrows=0).columns

prs_columns = [
    col for col in train_columns
    if col.startswith("ZSCORE_")
]

print("Number of PRS features:", len(prs_columns))

# 2. Assign data types to reduce memory usage
train_types = {col: "float32" for col in prs_columns}
test_types = {col: "float32" for col in prs_columns}

train_types["predicted_ancestry"] = "string"
test_types["predicted_ancestry"] = "string"

print("Loading training data...")
train = pd.read_csv("train.csv", dtype=train_types)

print("Loading test data...")
test = pd.read_csv("test.csv", dtype=test_types)

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

# 3. Separate model features from the target
features = ["predicted_ancestry"] + prs_columns

X = train[features]
y = train["breast_cancer"].astype(int)
X_test = test[features]

assert set(features).issubset(test_columns)
assert y.isin([0, 1]).all()

# 4. Define preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("prs", numeric_pipeline, prs_columns),
    ("ancestry", categorical_pipeline, ["predicted_ancestry"])
])

# 5. Create logistic regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        C=0.0003,
        solver="liblinear",
        max_iter=3000,
        random_state=42
    ))
])

# 6. Create a stratified validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 7. Train and evaluate
print("Training validation model...")
model.fit(X_train, y_train)

valid_probabilities = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, valid_probabilities)

print("Validation AUC:", round(auc, 5))

# 8. Retrain on all labeled samples
print("Training final model...")
model.fit(X, y)

# 9. Generate probabilities for Kaggle test data
print("Generating test predictions...")
predictions = model.predict_proba(X_test)[:, 1]

# 10. Save predictions with original test IDs
submission = pd.DataFrame({
    "id": test["id"],
    "breast_cancer": predictions
})

assert len(submission) == len(test)
assert submission["breast_cancer"].between(0, 1).all()

submission.to_csv("submission.csv", index=False)

print("Submission created successfully!")
print("Submission shape:", submission.shape)
print(submission.head())
