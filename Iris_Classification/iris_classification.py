# ==========================================================
# Supervised Classification on Iris Dataset
# Logistic Regression vs Decision Tree
# ==========================================================

# -----------------------------
# Import Libraries
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("iris.txt")

print("\nFirst 5 Records")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nClass Distribution")
print(df["species"].value_counts())

# -----------------------------
# 2. Exploratory Data Analysis
# -----------------------------
sns.pairplot(df, hue="species")
plt.show()

# -----------------------------
# 3. Encode Target Variable
# -----------------------------
encoder = LabelEncoder()

df["species"] = encoder.fit_transform(df["species"])

# -----------------------------
# 4. Split Features & Target
# -----------------------------
X = df.drop("species", axis=1)
y = df["species"]

# -----------------------------
# 5. Train-Test Split (80:20)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# 6. Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 7. Logistic Regression Model
# -----------------------------
lr = LogisticRegression(max_iter=200)

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

# -----------------------------
# 8. Decision Tree Model
# -----------------------------
dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

# -----------------------------
# 9. Evaluation Function
# -----------------------------
def evaluate_model(name, y_true, y_pred):

    print("\n" + "="*50)
    print(name)
    print("="*50)

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall   :", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score :", f1_score(y_true, y_pred, average="weighted"))

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))


evaluate_model(
    "Logistic Regression",
    y_test,
    y_pred_lr
)

evaluate_model(
    "Decision Tree",
    y_test,
    y_pred_dt
)

# -----------------------------
# 10. Confusion Matrix
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12,5))

cm1 = confusion_matrix(y_test, y_pred_lr)

sns.heatmap(
    cm1,
    annot=True,
    cmap="Blues",
    fmt="d",
    ax=ax[0]
)

ax[0].set_title("Logistic Regression")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

cm2 = confusion_matrix(y_test, y_pred_dt)

sns.heatmap(
    cm2,
    annot=True,
    cmap="Greens",
    fmt="d",
    ax=ax[1]
)

ax[1].set_title("Decision Tree")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# -----------------------------
# 11. Performance Comparison
# -----------------------------
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr, average="weighted"),
        precision_score(y_test, y_pred_dt, average="weighted")
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr, average="weighted"),
        recall_score(y_test, y_pred_dt, average="weighted")
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr, average="weighted"),
        f1_score(y_test, y_pred_dt, average="weighted")
    ]
})

print("\nPerformance Comparison")
print(results)

# -----------------------------
# 12. Decision Boundary
# -----------------------------
X = df.iloc[:, :2].values
y = df["species"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

model = LogisticRegression()

model.fit(X_train, y_train)

h = 0.02

x_min, x_max = X_train[:,0].min()-1, X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1, X_train[:,1].max()+1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(xx, yy, Z, alpha=0.3)

scatter = plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    edgecolors="k"
)

plt.xlabel("Sepal Length (Scaled)")
plt.ylabel("Sepal Width (Scaled)")
plt.title("Decision Boundary - Logistic Regression")

plt.legend(
    scatter.legend_elements()[0],
    encoder.classes_,
    title="Species"
)

plt.show()