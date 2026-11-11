
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# Create folders if they do not exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Generate sample student data
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "attendance": np.random.uniform(40, 100, n),
    "previous_marks": np.random.uniform(30, 100, n),
    "study_hours": np.random.uniform(0, 10, n),
    "assignment_completion": np.random.uniform(30, 100, n),
    "sleep_hours": np.random.uniform(3, 10, n)
})

# Generate final marks using a sample relationship
noise = np.random.normal(0, 5, n)

data["final_marks"] = (
    0.20 * data["attendance"]
    + 0.30 * data["previous_marks"]
    + 2.5 * data["study_hours"]
    + 0.15 * data["assignment_completion"]
    + 1.0 * data["sleep_hours"]
    + noise
).clip(0, 100)

# Save dataset
data.to_csv("data/student_performance.csv", index=False)

# Prepare features and target
X = data.drop("final_marks", axis=1)
y = data["final_marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train three ML models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions)
    })

# Compare models
results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv("model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Select and save the best model
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

joblib.dump(best_model, "models/student_performance_model.joblib")

print(f"\nBest model: {best_name}")
print("Model saved successfully!")
print("Dataset saved successfully!")
