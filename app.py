import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go


# Load model and scaler
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/saved_model.keras')

model = load_model()
scaler = joblib.load('model/scaler.save')

st.title("🏠 Boston Housing Price Predictor")

st.write("Enter house features below:")

# Create input fields for each feature
CRIM = st.number_input("CRIM (per capita crime rate)", 0.0, 100.0, 0.02)
ZN = st.number_input("ZN (residential land zoned %)", 0.0, 100.0, 18.0)
INDUS = st.number_input("INDUS (non-retail business acres)", 0.0, 30.0, 2.31)
CHAS = st.selectbox("CHAS (bounded by river)", [0, 1])
NOX = st.number_input("NOX (nitric oxides concentration)", 0.0, 1.0, 0.538)
RM = st.number_input("RM (average rooms per dwelling)", 1.0, 10.0, 6.575)
AGE = st.number_input("AGE (percent built before 1940)", 0.0, 100.0, 65.2)
DIS = st.number_input("DIS (distance to employment centers)", 0.0, 15.0, 4.09)
RAD = st.number_input("RAD (index of accessibility to highways)", 1.0, 24.0, 1.0)
TAX = st.number_input("TAX (property tax rate)", 100.0, 800.0, 296.0)
PTRATIO = st.number_input("PTRATIO (pupil-teacher ratio)", 10.0, 30.0, 15.3)
B = st.number_input("B (Black population index)", 0.0, 400.0, 396.9)
LSTAT = st.number_input("LSTAT (lower status population %)", 0.0, 40.0, 4.98)

# # Show radar/spider chart of input profile
feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                 "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

feature_values = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
                  DIS, RAD, TAX, PTRATIO, B, LSTAT]

# avg_values = [3.6, 11.4, 11.1, 0.07, 0.55, 6.3, 68.6,
#               3.8, 9.5, 408.2, 18.5, 356.7, 12.6]  # Rough dataset averages

# fig = go.Figure()

# fig.add_trace(go.Scatterpolar(
#     r=feature_values,
#     theta=feature_names,
#     fill='toself',
#     name='Your House'
# ))

# fig.add_trace(go.Scatterpolar(
#     r=avg_values,
#     theta=feature_names,
#     fill='toself',
#     name='Dataset Average'
# ))

# fig.update_layout(
#     polar=dict(
#         radialaxis=dict(visible=True),
#     ),
#     showlegend=True
# )

# st.plotly_chart(fig)

# Predict button
if st.button("Predict Price"):
    features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
                          DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    features_scaled = scaler.transform(features)
    pred = float(model.predict(features_scaled)[0][0])
    st.success(f"🏡 Predicted House Price: **${pred * 1000:.2f}**")

    # Feature importance approximation
    st.subheader("🔍 Top Feature Influence (by magnitude)")
    
    # Recreate the scaled features for importance estimation
    scaled_vals = scaler.transform(features)[0]
    abs_scaled = np.abs(scaled_vals)
    
    # Get top 5 features
    top_indices = abs_scaled.argsort()[::-1][:5]
    top_features = [feature_names[i] for i in top_indices]
    top_values = [scaled_vals[i] for i in top_indices]

    # Bar chart
    st.bar_chart(data=dict(zip(top_features, top_values)))

