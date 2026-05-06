import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("drowsiness_model.h5")

# Model class labels
classes = ["closed","no_yawn","open","yawn"]

st.title("Driver Drowsiness Detection System")

st.write("Upload driver images to detect fatigue level")

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg","png","jpeg"],
    accept_multiple_files=True
)

fatigue_levels = []
time_steps = []

if uploaded_files:

    for i, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Image {i+1}", width=250)

        # preprocessing
        img = image.resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)

        result = classes[class_index]

        st.write("Model Prediction:", result)

        # Decision Fusion Logic (Step 9)
        if result == "open" or result == "no_yawn":
            fatigue_stage = "Alert"
            level = 0

        elif result == "yawn":
            fatigue_stage = "Mild Fatigue"
            level = 1

        else:
            fatigue_stage = "Severe Fatigue"
            level = 2

        st.write("Fatigue Stage:", fatigue_stage)

        fatigue_levels.append(level)
        time_steps.append(i)

        st.write("---")

    # Fatigue Progression Curve (Step 10)
    st.subheader("Driver Fatigue Progression Curve")

    fig, ax = plt.subplots()

    ax.plot(time_steps, fatigue_levels, marker='o')

    ax.set_xlabel("Time (Image Sequence)")
    ax.set_ylabel("Fatigue Level")
    ax.set_title("Fatigue Progression Over Time")

    ax.set_yticks([0,1,2])
    ax.set_yticklabels(["Alert","Mild Fatigue","Severe Fatigue"])

    st.pyplot(fig)