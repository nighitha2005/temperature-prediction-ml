import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Create a sample temperature dataset
np.random.seed(42)

n = 500

humidity = np.random.randint(40, 95, n)
pressure = np.random.uniform(1000, 1025, n)
wind_speed = np.random.uniform(1, 15, n)
cloud_cover = np.random.randint(10, 100, n)

temperature = (
    35
    - 0.10 * humidity
    + 0.03 * (pressure - 1010)
    - 0.15 * wind_speed
    - 0.03 * cloud_cover
    + np.random.normal(0, 1.2, n)
)

df = pd.DataFrame({
    "Humidity": humidity,
    "Pressure": pressure,
    "Wind_Speed": wind_speed,
    "Cloud_Cover": cloud_cover,
    "Temperature": temperature
})

# 2. Save the dataset
df.to_csv("temperature_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())

# 3. Select input features and target
X = df[[
    "Humidity",
    "Pressure",
    "Wind_Speed",
    "Cloud_Cover"
]]

y = df["Temperature"]

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results")
print("------------------------")
print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2 Score : {r2:.3f}")

# 8. Save the trained model
joblib.dump(model, "temperature_prediction_model.pkl")

print("\nModel saved successfully!")
print("File: temperature_prediction_model.pkl")