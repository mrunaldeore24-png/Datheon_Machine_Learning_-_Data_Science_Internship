import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("data.txt")

print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

# ---------------------------------------------------
# Handle Missing Values
# ---------------------------------------------------

num_cols = df.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy="median")
df[num_cols] = imputer.fit_transform(df[num_cols])

# ---------------------------------------------------
# Encode Categorical Feature
# ---------------------------------------------------

df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# ---------------------------------------------------
# Prepare Data
# ---------------------------------------------------

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ---------------------------------------------------
# Standardization
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# Linear Regression
# ---------------------------------------------------

ols = LinearRegression()
ols.fit(X_train, y_train)

ols_pred = ols.predict(X_test)

# ---------------------------------------------------
# Ridge Regression
# ---------------------------------------------------

ridge = GridSearchCV(
    Ridge(),
    {"alpha":[0.01,0.1,1,10,100]},
    cv=5
)

ridge.fit(X_train,y_train)

ridge_pred = ridge.predict(X_test)

print("\nBest Ridge Alpha:", ridge.best_params_)

# ---------------------------------------------------
# Lasso Regression
# ---------------------------------------------------

lasso = GridSearchCV(
    Lasso(max_iter=5000),
    {"alpha":[0.001,0.01,0.1,1]},
    cv=5
)

lasso.fit(X_train,y_train)

lasso_pred = lasso.predict(X_test)

print("Best Lasso Alpha:", lasso.best_params_)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

def evaluate(name,y_true,y_pred):

    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    mae = mean_absolute_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)

    print("\n",name)
    print("RMSE :",rmse)
    print("MAE  :",mae)
    print("R2   :",r2)

evaluate("Linear Regression",y_test,ols_pred)
evaluate("Ridge Regression",y_test,ridge_pred)
evaluate("Lasso Regression",y_test,lasso_pred)

# ---------------------------------------------------
# Comparison Table
# ---------------------------------------------------

results = pd.DataFrame({

    "Model":["Linear","Ridge","Lasso"],

    "RMSE":[
        np.sqrt(mean_squared_error(y_test,ols_pred)),
        np.sqrt(mean_squared_error(y_test,ridge_pred)),
        np.sqrt(mean_squared_error(y_test,lasso_pred))
    ],

    "MAE":[
        mean_absolute_error(y_test,ols_pred),
        mean_absolute_error(y_test,ridge_pred),
        mean_absolute_error(y_test,lasso_pred)
    ],

    "R2":[
        r2_score(y_test,ols_pred),
        r2_score(y_test,ridge_pred),
        r2_score(y_test,lasso_pred)
    ]

})

print("\nPerformance Comparison")
print(results)

# ---------------------------------------------------
# Predicted vs Actual
# ---------------------------------------------------

plt.figure(figsize=(7,6))

plt.scatter(y_test,ridge_pred,alpha=0.5)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual (Ridge)")

plt.plot(
    [y_test.min(),y_test.max()],
    [y_test.min(),y_test.max()],
    'r--'
)

plt.show()

# ---------------------------------------------------
# Residual Distribution
# ---------------------------------------------------

residuals = y_test-ridge_pred

plt.figure(figsize=(7,5))

sns.histplot(residuals,kde=True)

plt.title("Residual Distribution")

plt.show()