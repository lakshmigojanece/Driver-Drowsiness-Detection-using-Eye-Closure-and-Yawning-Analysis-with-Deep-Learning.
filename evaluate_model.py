import tensorflow as tf #load model(deep learning) & run 
from tensorflow.keras.preprocessing.image import ImageDataGenerator #used for imageprocesssing and augmentaion
from sklearn.metrics import classification_report, confusion_matrix #evaluate model performance
import numpy as np #array operation

IMG_SIZE = (224,224)

model = tf.keras.models.load_model("drowsiness_model.h5")#already trained model so that we can use reuse without retraining

#testing Data (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

#Load testing Data
test_generator = test_datagen.flow_from_directory(
"dataset/test", #test image folder
target_size=IMG_SIZE, #resize images
batch_size=32, #32 images at a time 
class_mode="categorical",#multi class labels
shuffle=False #avoild mismatch classes
)

loss,acc = model.evaluate(test_generator)#model performance calculate error and accuracy

print("Test Accuracy:",acc) #final accuracy--correct prediction/total pred, it is used to check overall performance

predictions = model.predict(test_generator)#model output probabilties
y_pred = np.argmax(predictions,axis=1)#highest value index

print("Confusion Matrix")# Actual vs predicted compare ,how to miskake model show
print(confusion_matrix(test_generator.classes,y_pred))

print("Classification Report")# precision--TP/(TP+FP)--correct positive show, recall--dectect actual positives --TP/(TP+FN), 
#f1-score--balanced performance measure--2x(precision x recall)/(precision + recall)
print(classification_report(test_generator.classes,y_pred))
