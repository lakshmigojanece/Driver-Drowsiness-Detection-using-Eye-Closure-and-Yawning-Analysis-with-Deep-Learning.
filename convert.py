from tensorflow.keras.models import load_model

model = load_model("drowsiness_model.h5", compile=False)
model.save("new_model.keras")

print("DONE")