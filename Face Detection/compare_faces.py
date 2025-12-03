import numpy as np
import tensorflow as tf
from keras.src.utils import load_img, img_to_array
import os

class CompareFaces:
    def __init__(self, model_path, img_size=(100, 100), threshold=0.5):
        self.model_path = model_path
        self.img_size = img_size
        self.threshold = threshold

        print(f"[INFO] Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("[INFO] Model loaded successfully!")

    def preprocess(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Must match training: RGB, 100×100
        img = load_img(img_path, target_size=self.img_size, color_mode="rgb")
        img = img_to_array(img) / 255.0
        return np.expand_dims(img, axis=0)

    def get_embedding(self, img_path):
        img = self.preprocess(img_path)
        embedding = self.model.predict(img)
        return embedding

    def compare_faces(self, img1_path, img2_path):
        emb1 = self.get_embedding(img1_path)
        emb2 = self.get_embedding(img2_path)

        # Euclidean distance
        distance = np.linalg.norm(emb1 - emb2)
        is_match = distance < self.threshold

        return float(distance), is_match