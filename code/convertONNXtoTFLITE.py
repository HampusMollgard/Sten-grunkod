import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load ONNX model
onnx_model = onnx.load("model.onnx")

# Convert ONNX model to TensorFlow model
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_saved_model")

# Convert TensorFlow model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model_saved_model")
tflite_model = converter.convert()

# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)