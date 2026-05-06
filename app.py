#import libraries
import streamlit as st #web app create 
import tensorflow as tf 
import numpy as np
import cv2 #image precessing--(face,eye detect)
from PIL import Image # read image
import matplotlib.pyplot as plt #graph

st.title("🚗 Driver Drowsiness Detection System")

model = tf.keras.models.load_model("drowsiness_model.h5") # used for already trained model

#classes names
class_names = ["closed","no_yawn","open","yawn"]

#Face & eye detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
) # already trained use to learn the patterns 

eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

#file upload UI
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

#store Fatigue Scores
fatigue_scores = []#each frame/image fatigue level store

#preprocessing Function
def preprocess(img):

    img = cv2.resize(img,(224,224))
    img = img / 255.0
    img = np.expand_dims(img,axis=0)

    return img

#fatigue logic function 
def fatigue_stage(pred):

    if pred in ["open","no_yawn"]:
        return "🟢 Alert",0

    elif pred == "yawn":
        return "🟡 Mild Fatigue",1

    else:
        return "🔴 Severe Fatigue",2

#main loop
if uploaded_files: #if user upload images --start processing

    for file in uploaded_files:#one by one image process

        image = Image.open(file).convert("RGB") # read images--numpy format convert
        img = np.array(image)

        st.image(img,width=300)#show images

# Face detection
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #images convert into grayscale , detect face black and white only brightness 

        faces = face_detector.detectMultiScale(gray,1.3,5)

        roi = img # default image,select images in important part only (region of interest)

        #face found
        if len(faces)>0:

            x,y,w,h = faces[0]
            face = img[y:y+h,x:x+w] # face crop , x-left position, y--right,w--width, h--height

            eyes = eye_detector.detectMultiScale(
                cv2.cvtColor(face,cv2.COLOR_BGR2GRAY),
                1.2,
                4
            ) # eyes detector
# eye found
            if len(eyes)>0:

                ex,ey,ew,eh = eyes[0] #eye detection 
                roi = face[ey:ey+eh , ex:ex+ew]

                st.success("Eye detected")

            else:

                roi = face #full face
                st.warning("Eye not detected → using face")

        else:

            st.info("No face detected → assuming cropped eye/mouth image")# if no face

# prediction
        input_img = preprocess(roi)

        pred = model.predict(input_img)

#class
        class_id = np.argmax(pred)

        label = class_names[class_id]
#confidence
        confidence = float(np.max(pred))

# fatigue stage
        stage,score = fatigue_stage(label) # convert to alert level
#store score
        fatigue_scores.append(score)

        st.write("Prediction:",label)
        st.write("Confidence:",round(confidence,3))
        st.write("Fatigue Stage:",stage)

        st.write("---")

    if fatigue_scores:

        st.subheader("Driver Fatigue Progression Curve")

        plt.figure()

        plt.plot(fatigue_scores,marker="o")

        plt.yticks(
            [0,1,2],
            ["Alert","Mild Fatigue","Severe Fatigue"]
        )

        plt.xlabel("Frame")
        plt.ylabel("Fatigue Level")

        st.pyplot(plt)