import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="Driver Drowsiness Detection Demo", layout="wide")
st.title("🧠 Driver Drowsiness Detection (Demo Mode)")

st.info("⚠ Demo Mode: Model not loaded. Predictions are random for testing the app UI.")

# -------------------------------
# Fatigue Mapping
# -------------------------------
def fatigue_stage(pred_class):
    if pred_class in ['Open', 'no_yawn']:
        return "Alert", 0
    elif pred_class == 'yawn':
        return "Mild Fatigue", 1
    elif pred_class == 'Closed':
        return "Severe Fatigue", 2

# -------------------------------
# Fake Prediction Function
# -------------------------------
def predict_image_demo(img):
    classes = ['Closed', 'Open', 'no_yawn', 'yawn']
    pred_class = random.choice(classes)
    stage_label, stage_num = fatigue_stage(pred_class)
    return pred_class, stage_label, stage_num

# -------------------------------
# Upload Multiple Images
# -------------------------------
st.header("📸 Upload Driver Images")
uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

all_stages = []

if uploaded_files:
    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=file.name)

        pred_class, stage_label, stage_num = predict_image_demo(img)

        st.success(f"Prediction: {pred_class}")
        st.warning(f"Fatigue: {stage_label}")

        all_stages.append(stage_num)

# -------------------------------
# Fatigue Progression Curve
# -------------------------------
st.header("📈 Fatigue Progression Curve")

if len(all_stages) > 0:
    interval = st.slider("Frames per interval", 1, 10, 3)

    avg_stages = [
        np.mean(all_stages[i:i+interval])
        for i in range(0, len(all_stages), interval)
    ]

    time_intervals = list(range(len(avg_stages)))

    # Plot graph
    plt.figure(figsize=(10, 4))
    plt.plot(time_intervals, avg_stages, marker='o')
    plt.yticks([0, 1, 2], ["Alert", "Mild Fatigue", "Severe Fatigue"])
    plt.xlabel("Time Interval")
    plt.ylabel("Fatigue Level")
    plt.title("Driver Fatigue Progression Curve (Demo)")
    plt.grid(True)

    st.pyplot(plt)

    # Interpretation
    st.subheader("📊 Interpretation")
    for i, val in enumerate(avg_stages):
        if val < 0.5:
            st.write(f"Interval {i}: Alert")
        elif val < 1.5:
            st.write(f"Interval {i}: Mild Fatigue")
        else:
            st.write(f"Interval {i}: Severe Fatigue")
else:
    st.info("Upload images to see fatigue progression")