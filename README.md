# 🏠 Boston Housing Price Predictor (ML + Streamlit)

This project uses a trained regression model built with TensorFlow to predict housing prices in Boston based on 13 numeric features such as crime rate, average number of rooms, tax level, and distance to employment centers. It supports both real-time predictions via a web UI and batch predictions through CSV file processing.

🔗 **Live Web App**  
[https://boston-housing-predictor-0khat0.streamlit.app](https://boston-housing-predictor-0khat0.streamlit.app)

---

## ✨ Key Features

- 🔮 **Real-time price prediction** through a user-friendly web interface
- 📥 **Batch mode prediction** for multiple rows using CSV input
- 📉 Bar chart of top influencing features (based on scaled input values)
- 🧠 Trained neural network using TensorFlow/Keras
- ☁️ Fully deployed on Streamlit Cloud

---

## ⚙️ Tech Stack

| Component      | Tools                                      |
|----------------|--------------------------------------------|
| Modeling       | TensorFlow, Keras                          |
| Preprocessing  | scikit-learn `StandardScaler`, pandas      |
| Interface      | Streamlit                                  |
| Visualization  | Plotly                                     |
| Deployment     | Streamlit Community Cloud                  |
| Code Hosting   | GitHub                                     |

---

## 📁 Project Structure

```plaintext
├── app.py                 # Main Streamlit application
├── model/
│   ├── saved_model.keras  # Trained TensorFlow model
│   └── scaler.save        # Fitted StandardScaler
├── requirements.txt       # All required Python dependencies
├── README.md
├── training/              # Development scripts and utilities
│   ├── train.py           # Model training
│   ├── predict.py         # Command-line prediction for one input
│   ├── batch_predict.py   # Batch CSV prediction
│   ├── main.py            # Optional FastAPI interface (not deployed)
│   └── data/
│       ├── boston.csv             # Source dataset
│       ├── batch_input.csv        # Example batch input
│       └── predicted_output.csv   # Output from batch prediction
