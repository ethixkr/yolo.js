import tensorflow as tf
from keras.models import load_model


model = load_model("model.h5")
# Create a converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Convert the model
tflite_model = converter.convert()
# Create the tflite model file
tflite_model_name = "mymodel.tflite"
open(tflite_model_name, "wb").write(tflite_model)
