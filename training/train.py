import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

# Load dataset
df = pd.read_csv('data/boston.csv')

# Features and target
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, verbose=1)

# Evaluate on test set
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Test MAE: {mae:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
model.save('model/saved_model.keras')

# Also save the scaler for predictions
import joblib
joblib.dump(scaler, 'model/scaler.save')
