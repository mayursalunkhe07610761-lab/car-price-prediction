import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("car data.csv")

# Create car age feature
data['Car_Age'] = 2026 - data['Year']

# Keep important columns
data = data[['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age', 'Selling_Price']]

# Features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)

print("R2 Score:", score)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
pickle.dump(model, open("models/car_model.pkl", "wb"))

print("Model saved successfully")