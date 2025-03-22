import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("updated_salary_data.csv")

# Encode categorical columns
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df = pd.get_dummies(df, columns=["Education"], drop_first=True)

# Define features & target
X = df.drop(columns=["Salary"])
y = df["Salary"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model (Random Forest for better performance)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "salary_model1.pkl")

print("✅ Model and Scaler saved as salary_model.pkl and scaler.pkl")
