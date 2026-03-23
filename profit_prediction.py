"""
ProfitIQ — Startup Profit Prediction
Author: Swapnil Chitalkar

Predicts startup profit based on R&D Spend, Administration,
and Marketing Spend using multiple regression models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD DATA ──────────────────────────────────────────
try:
    df = pd.read_csv("50_Startups.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: '50_Startups.csv' not found. Please ensure the file is in the same directory as this script.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# ── 2. PREPROCESSING ──────────────────────────────────────
# Check for required columns
required_columns = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns: {missing_columns}")
    exit(1)

# Drop non-numeric State column if present
if "State" in df.columns:
    df = df.drop("State", axis=1)

X = df.drop("Profit", axis=1)
y = df["Profit"]

print("\nFeatures:", list(X.columns))
print("Target: Profit")

# ── 3. CORRELATION ANALYSIS ───────────────────────────────
print("\n── Correlation with Profit ──")
print(df.corr()["Profit"].sort_values(ascending=False))

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("Saved: correlation_heatmap.png")

# ── 4. TRAIN-TEST SPLIT ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 5. TRAIN MODELS ───────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = []

print("\n── Model Performance ──")
print(f"{'Model':<25} {'R² Score':>10} {'MAE':>12} {'RMSE':>12}")
print("-" * 62)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
    print(f"{name:<25} {r2:>10.4f} {mae:>12.2f} {rmse:>12.2f}")

# ── 6. BEST MODEL ─────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
best_model_name = results_df.iloc[0]["Model"]
best_r2 = results_df.iloc[0]["R2"]

print(f"\n✅ Best Model: {best_model_name} with R² = {best_r2:.4f}")

# ── 7. VISUALIZE RESULTS ──────────────────────────────────
plt.figure(figsize=(10, 5))
colors = ["#2ecc71" if m == best_model_name else "#3498db" for m in results_df["Model"]]
plt.barh(results_df["Model"], results_df["R2"], color=colors)
plt.xlabel("R² Score")
plt.title("Model Comparison — R² Score")
plt.xlim(0, 1.05)
for i, v in enumerate(results_df["R2"]):
    plt.text(v + 0.01, i, f"{v:.4f}", va="center")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()
print("Saved: model_comparison.png")

# ── 8. FEATURE IMPORTANCE (Random Forest) ─────────────────
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\n── Feature Importance (Random Forest) ──")
print(importances)

plt.figure(figsize=(7, 4))
importances.plot(kind="bar", color="#1F4E79")
plt.title("Feature Importance — Random Forest")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("Saved: feature_importance.png")

print("\n── Done! All outputs saved. ──")
