import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import json

IMG_SIZE = (224,224) #input image size-->mobilenet expects 224X224
BATCH_SIZE = 32 #model processes 32 images at a time

#training Data (Augmentation + Rescaling)
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=30,
zoom_range=0.2,
horizontal_flip=True,
width_shift_range=0.2,
height_shift_range=0.2,
brightness_range=[0.5,1.5],
shear_range=0.2
)
#validation Data (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)


#Load Training Data
train_generator = train_datagen.flow_from_directory(
"dataset/train",
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode="categorical"
)
# load validation Data
val_generator = val_datagen.flow_from_directory(
"dataset/val",
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode="categorical"
)
#load pretrained model(MobileNetV2)
base_model = MobileNetV2( # already train model donot retrain model
weights="imagenet", # used pretrained weights
include_top=False, #remove last classification layers and there are using four classes
input_shape=(224,224,3)
)
#Freeze Layers
for layer in base_model.layers: #donot update model weights
    layer.trainable = False #freeze concepts
#Add Custom layers
x = base_model.output #features
x = GlobalAveragePooling2D()(x) #big feature is convert into single vector
x = Dense(128,activation="relu")(x) #hidden layers,learn model, 128 neurons, decesion making
predictions = Dense(4,activation="softmax")(x) #final output, 4--> classes convert probability

# create Model
model = Model(inputs=base_model.input, outputs=predictions) #built full model,input->image,output->predictions, base model+new layers combine 

#Compile model
model.compile(
optimizer="adam", #update weights
loss="categorical_crossentropy",#error calculate,use multi-class classification
metrics=["accuracy"] #measure performances,check correct predict
)

#Train Model
history = model.fit(
train_generator,#learning
validation_data=val_generator,#checking
epochs=50
)

#save model
model.save("drowsiness_model.h5")

#Save class labels
with open("class_indices.json","w") as f:
    json.dump(train_generator.class_indices,f)

#plot Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])
plt.show()
