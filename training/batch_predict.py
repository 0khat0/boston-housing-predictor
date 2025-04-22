import tensorflow as tf
import joblib
import pandas as pd

# Load model and scaler
model = tf.keras.models.load_model('model/saved_model.keras')
scaler = joblib.load('model/scaler.save')

# Load batch data (no MEDV column)
df = pd.read_csv('data/batch_input.csv')

# Scale features
X_scaled = scaler.transform(df)

# Predict
predictions = model.predict(X_scaled)

# Append predictions to DataFrame
df['Predicted_MEDV'] = predictions.flatten()

# Save to CSV
df.to_csv('data/predicted_output.csv', index=False)

print("✅ Predictions saved to data/predicted_output.csv")
