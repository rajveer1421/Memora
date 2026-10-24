import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf, tf2onnx, numpy as np
from deepface.models.facial_recognition.ArcFace import load_model
m = load_model()  # uses cached ~/.deepface/weights/arcface_weights.h5
spec = (tf.TensorSpec((None, 112, 112, 3), tf.float32, name="input"),)
tf2onnx.convert.from_keras(m, input_signature=spec, opset=13, output_path=sys.argv[1])
x = np.random.default_rng(0).random((2, 112, 112, 3), dtype=np.float32)
ref = m.predict(x, verbose=0)
import onnxruntime as ort
out = ort.InferenceSession(sys.argv[1]).run(None, {"input": x})[0]
print("max abs diff keras vs onnx:", float(np.abs(ref - out).max()), "shape", out.shape)
