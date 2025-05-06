import streamlit as st
import numpy as np
import os
import joblib
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# === Title ===
st.title("ANN Model for Frequency Prediction of Plates with Cutouts")

# === Utility Functions ===

# Safely load a Keras model
def load_model_safe(filepath):
    try:
        return tf.keras.models.load_model(filepath)
    except Exception:
        return None

# Safely load a StandardScaler
def load_scaler_safe(shape, boundary):
    scaler_path = f"scaler_{shape.lower()}_{boundary}.pkl"
    try:
        return joblib.load(scaler_path)
    except FileNotFoundError:
        return None

# Perform prediction
def make_prediction(model, scaler, input_data):
    scaled = scaler.transform([input_data])
    return model.predict(scaled)[0][0]

# === Sidebar Inputs ===

st.sidebar.header("Select Parameters")
cutout_shape = st.sidebar.selectbox("Cutout Shape", ["Circle", "Square", Capsule])
boundary_condition = st.sidebar.selectbox("Boundary Condition", ["CCCC", "CCCF", "CCFF", "CFFF", "SSSS", "CFCF"])

# Plate dimensions
length = st.sidebar.number_input("Length (mm)", min_value=0.0, value=100.0, step=1.0)
breadth = st.sidebar.number_input("Breadth (mm)", min_value=0.0, value=100.0, step=1.0)
thickness = st.sidebar.number_input("Thickness (mm)", min_value=0.0, value=1.0, step=1.0)

# Cutout clearance rule
clearance = 3.0
input_data = None
valid = True
error_messages = []

# Shape-specific inputs
if cutout_shape == "Circle":
    diameter = st.sidebar.number_input("Diameter (mm)", min_value=0.0, value=10.0, step=1.0)
    input_data = np.array([length, breadth, diameter, thickness])
    if (diameter + 2 * clearance > length) or (diameter + 2 * clearance > breadth) :
        error_messages.append("Circle cutout (with 3mm edge clearance) does not fit within the plate.")
        valid = False

elif cutout_shape == "Square":
    side = st.sidebar.number_input("Side (mm)", min_value=0.0, value=10.0, step=1.0)
    input_data = np.array([length, breadth, side, thickness])
    if (side + 2 * clearance > length) or (side + 2 * clearance > breadth) :
        error_messages.append("Square cutout (with 3mm edge clearance) does not fit within the plate.")
        valid = False

elif cutout_shape == "Capsule":
    diameter = st.sidebar.number_input("Diameter (mm)", min_value=0.0, value=10.0, step=1.0)
    side = st.sidebar.number_input("Side (mm)", min_value=0.0, value=10.0, step=1.0)
    input_data = np.array([length, breadth, diameter, side, thickness])
    if (diameter + 2 * clearance > length) or (side + 2 * clearance > breadth):
        error_messages.append("Capsule cutout (with 3mm edge clearance) does not fit within the plate.")
        valid = False

# Show any dimension validation errors
for msg in error_messages:
    st.error(f"❌ {msg}")

# === Model and Prediction ===

predict_clicked = st.sidebar.button("Predict")

if predict_clicked:
    if not valid:
        st.error("Please adjust input dimensions so the cutout fits within the plate with 3mm edge clearance.")
    else:
        # Build model paths
        shape_key = cutout_shape.lower()
        bc_key = boundary_condition

        model_dir = "models"
        model_paths = [
            f"{shape_key}_{bc_key}_f{i}.keras" for i in range(1, 5)
        ]
        shape_models = [load_model_safe(path) for path in model_paths]
        scaler = load_scaler_safe(cutout_shape, boundary_condition)

        if any(m is None for m in shape_models) or scaler is None:
            st.error(f"❌ One or more models or the scaler are missing for {cutout_shape} with {boundary_condition}.")
        else:
            try:
                predictions = [make_prediction(model, scaler, input_data) for model in shape_models]
                st.success(f"✅ Predictions for {cutout_shape} cutout with {boundary_condition} boundary:")
                for i, freq in enumerate(predictions, 1):
                    st.write(f"**Predicted Frequency F{i}**: {freq:.4f} Hz")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


