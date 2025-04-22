import tensorflow as tf
import joblib
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model/saved_model.keras')

# Load the scaler
scaler = joblib.load('model/scaler.save')

# Sample input
sample_input = np.array([[0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]])

# Scale input
sample_input_scaled = scaler.transform(sample_input)

# Predict
predicted_price = model.predict(sample_input_scaled)

# Show result
print(f"Predicted house price: ${predicted_price[0][0]*1000:.2f}")
