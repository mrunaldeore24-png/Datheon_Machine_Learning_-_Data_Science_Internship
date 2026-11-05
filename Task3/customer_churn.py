# ==========================================================
# Customer Churn Prediction
# Random Forest vs XGBoost
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("Telco-Customer-Churn.txt")

print(df.head())

print("\nShape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ---------------------------------------------------
# Convert TotalCharges
# ---------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# ---------------------------------------------------
# Handle Missing Values
# ---------------------------------------------------

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ---------------------------------------------------
# Drop Customer ID
# ---------------------------------------------------

df.drop("customerID", axis=1, inplace=True)

# ---------------------------------------------------
# Encode Target
# ---------------------------------------------------

df["Churn"] = df["Churn"].map({
    "No":0,
    "Yes":1
})

# ---------------------------------------------------
# Features & Target
# ---------------------------------------------------

X = df.drop("Churn", axis=1)

y = df["Churn"]

# ---------------------------------------------------
# Categorical & Numeric Columns
# ---------------------------------------------------

categorical = X.select_dtypes(include="object").columns

numeric = X.select_dtypes(exclude="object").columns

# ---------------------------------------------------
# Preprocessing
# ---------------------------------------------------

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            Pipeline([
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
            ]),
            numeric
        ),

        (
            "cat",
            Pipeline([
                ("encoder",OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        )

    ]

)

# ---------------------------------------------------
# Transform Features
# ---------------------------------------------------

X_processed = preprocessor.fit_transform(X)

# ---------------------------------------------------
# Balance Classes
# ---------------------------------------------------

smote = SMOTE(random_state=42)

X_balanced, y_balanced = smote.fit_resample(
    X_processed,
    y
)

# ---------------------------------------------------
# Train-Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X_balanced,
    y_balanced,

    test_size=0.20,

    random_state=42

)

# ---------------------------------------------------
# Random Forest
# ---------------------------------------------------

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(X_train,y_train)

rf_prob = rf.predict_proba(X_test)[:,1]

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)

print("\nRandom Forest ROC-AUC:", rf_auc)

# ---------------------------------------------------
# XGBoost
# ---------------------------------------------------

xgb = XGBClassifier(

    eval_metric="logloss",

    random_state=42

)

xgb.fit(X_train,y_train)

xgb_prob = xgb.predict_proba(X_test)[:,1]

xgb_auc = roc_auc_score(
    y_test,
    xgb_prob
)

print("XGBoost ROC-AUC:", xgb_auc)

# ---------------------------------------------------
# Cross Validation
# ---------------------------------------------------

rf_cv = cross_val_score(

    rf,

    X_balanced,

    y_balanced,

    cv=5,

    scoring="roc_auc"

)

xgb_cv = cross_val_score(

    xgb,

    X_balanced,

    y_balanced,

    cv=5,

    scoring="roc_auc"

)

print("\nRandom Forest CV ROC-AUC:", rf_cv.mean())

print("XGBoost CV ROC-AUC:", xgb_cv.mean())

# ---------------------------------------------------
# Classification Report
# ---------------------------------------------------

print("\nRandom Forest Report")

print(classification_report(
    y_test,
    rf.predict(X_test)
))

print("\nXGBoost Report")

print(classification_report(
    y_test,
    xgb.predict(X_test)
))

# ---------------------------------------------------
# ROC Curve
# ---------------------------------------------------

rf_fpr,rf_tpr,_ = roc_curve(
    y_test,
    rf_prob
)

xgb_fpr,xgb_tpr,_ = roc_curve(
    y_test,
    xgb_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC={rf_auc:.3f})"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost (AUC={xgb_auc:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    '--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

importance = pd.Series(

    xgb.feature_importances_,

    index=feature_names

)

top10 = importance.sort_values(
    ascending=False
).head(10)

print("\nTop 10 Important Features")

print(top10)

plt.figure(figsize=(10,6))

top10.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Feature Importances")

plt.show()