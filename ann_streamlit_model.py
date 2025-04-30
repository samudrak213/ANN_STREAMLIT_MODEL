import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

# Title
st.title("ANN Model for Frequency Prediction of Plates with Cutouts")

# Simulate model loading (only load if the file exists)
def load_model_safe(filepath):
    return tf.keras.models.load_model(filepath) if os.path.exists(filepath) else None

# Define available models: {shape: {boundary: [model_f1, model_f2, model_f3, model_f4]}}
models = {
    "Circle": {
        "CCCC": [
            load_model_safe("ann_model1.keras"),
            load_model_safe("ann_model2.keras"),
            load_model_safe("ann_model3.keras"),
            load_model_safe("ann_model4.keras"),
        ]
    },
    "Square": {
        "CCCC": [
            load_model_safe("models/square_CCCC_f1.keras"),
            load_model_safe("models/square_CCCC_f2.keras"),
            load_model_safe("models/square_CCCC_f3.keras"),
            load_model_safe("models/square_CCCC_f4.keras"),
        ]
    },
    "Capsule": {
        "CCCC": [
            load_model_safe("models/capsule_CCCC_f1.keras"),
            load_model_safe("models/capsule_CCCC_f2.keras"),
            load_model_safe("models/capsule_CCCC_f3.keras"),
            load_model_safe("models/capsule_CCCC_f4.keras"),
        ]
    },
}


scaler = joblib.load('scaler.pkl')

# Sidebar selections
st.sidebar.header("Select Parameters")
cutout_shape = st.sidebar.selectbox("Cutout Shape", ["Circle", "Square", "Capsule"])
boundary_condition = st.sidebar.selectbox("Boundary Condition", ["CCCC"])

# Base inputs
length = st.sidebar.number_input("Length", min_value=0.0, value=1.0)
breadth = st.sidebar.number_input("Breadth", min_value=0.0, value=1.0)
thickness = st.sidebar.number_input("Thickness", min_value=0.0, value=1.0)

# Shape-specific inputs
input_data = None
if cutout_shape == "Circle":
    diameter = st.sidebar.number_input("Diameter", min_value=0.0, value=1.0)
    input_data = np.array([length, breadth, diameter, thickness])
elif cutout_shape == "Square":
    side = st.sidebar.number_input("Side", min_value=0.0, value=1.0)
    input_data = np.array([length, breadth, side, thickness])
elif cutout_shape == "Capsule":
    diameter = st.sidebar.number_input("Diameter", min_value=0.0, value=1.0)
    side = st.sidebar.number_input("Side", min_value=0.0, value=1.0)
    input_data = np.array([length, breadth, diameter, side, thickness])

# Prediction function
def make_prediction(model, input_data):
    scaled = scaler.fit_transform([input_data])
    return model.predict(scaled)[0][0]

# Predict button
if st.sidebar.button("Predict"):
    shape_models = models.get(cutout_shape, {}).get(boundary_condition, [None]*4)

    if any(m is None for m in shape_models):
        st.error(f"One or more models are missing for {cutout_shape} with {boundary_condition}.")
    else:
        try:
            preds = [make_prediction(m, input_data) for m in shape_models]
            st.write(f"### Predictions for {cutout_shape} cutout with {boundary_condition} boundary:")
            for i, f in enumerate(preds, start=1):
                st.write(f"**Predicted Frequency F{i}**: {f:.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
