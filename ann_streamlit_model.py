import streamlit as st
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

# Title
st.title("ANN Model for Frequency Prediction of Plates with Cutouts")

# Utility to safely load models
def load_model_safe(filepath):
    return tf.keras.models.load_model(filepath) if os.path.exists(filepath) else None

# Load models for each shape and boundary (only CCCC in this case)
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
    }
}

# Load scalers for each shape
scalers = {
    "Circle": joblib.load("scaler_circle_CCCC.pkl"),
    "Square": joblib.load("scaler_circle_CCCC.pkl"),
    "Capsule": joblib.load("scaler_circle_CCCC.pkl"),
}

# Sidebar inputs
st.sidebar.header("Select Parameters")
cutout_shape = st.sidebar.selectbox("Cutout Shape", ["Circle", "Square", "Capsule"])
boundary_condition = st.sidebar.selectbox("Boundary Condition", ["CCCC"])

# Plate dimensions
length = st.sidebar.number_input("Length (mm)", min_value=0.0, value=100.0, step=1.0)
breadth = st.sidebar.number_input("Breadth (mm)", min_value=0.0, value=100.0, step=1.0)
thickness = st.sidebar.number_input("Thickness (mm)", min_value=0.0, value=1.0, step=1.0)

# Cutout dimensions
input_data = None
valid = True
clearance = 3.0  # minimum clearance between plate edge and cutout

if cutout_shape == "Circle":
    diameter = st.sidebar.number_input("Diameter (mm)", min_value=0.0, value=10.0, step=1.0)
    input_data = np.array([length, breadth, diameter, thickness])
    if diameter + 2 * clearance > length or diameter + 2 * clearance > breadth:
        st.error("❌ Circle cutout (with 3mm edge clearance) does not fit within the plate.")
        valid = False

elif cutout_shape == "Square":
    side = st.sidebar.number_input("Side (mm)", min_value=0.0, value=10.0, step=1.0)
    input_data = np.array([length, breadth, side, thickness])
    if side + 2 * clearance > length or side + 2 * clearance > breadth:
        st.error("❌ Square cutout (with 3mm edge clearance) does not fit within the plate.")
        valid = False

elif cutout_shape == "Capsule":
    diameter = st.sidebar.number_input("Diameter (mm)", min_value=0.0, value=10.0, step=1.0)
    side = st.sidebar.number_input("Side (mm)", min_value=0.0, value=10.0, step=1.0)
    input_data = np.array([length, breadth, diameter, side, thickness])
    if diameter + 2 * clearance > length or side + 2 * clearance > breadth:
        st.error("❌ Capsule cutout (with 3mm edge clearance) does not fit within the plate.")
        valid = False

# Prediction logic
def make_prediction(model, scaler, input_data):
    scaled = scaler.transform([input_data])
    return model.predict(scaled)[0][0]

if valid and st.sidebar.button("Predict"):
    shape_models = models.get(cutout_shape, {}).get(boundary_condition, [None]*4)
    scaler = scalers.get(cutout_shape)

    if any(m is None for m in shape_models) or scaler is None:
        st.error(f"❌ One or more models or scaler are missing for {cutout_shape} with {boundary_condition}.")
    else:
        try:
            predictions = [make_prediction(m, scaler, input_data) for m in shape_models]
            st.success(f"✅ Predictions for {cutout_shape} cutout with {boundary_condition} boundary:")
            for i, freq in enumerate(predictions, 1):
                st.write(f"**Predicted Frequency F{i}**: {freq:.4f} Hz")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

