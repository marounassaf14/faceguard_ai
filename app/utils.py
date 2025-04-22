import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model

# Load FaceNet model globally for reuse
facenet_model = load_model("models/facenet_finetuned.h5")

# -------------------------------
# XCEPTION PREPROCESSING
# -------------------------------
def preprocess_xception(image_np):
    """
    Prepares a face image for Xception input (299x299 with preprocessing).
    """
    img = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.xception.preprocess_input(img)
    return tf.expand_dims(img, axis=0)  # [1, 299, 299, 3]

# -------------------------------
# FACENET PREPROCESSING
# -------------------------------
def preprocess_facenet(img_pil):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img_pil).unsqueeze(0)  # Add batch dim