import insightface
import numpy as np
import cv2

import insightface
import numpy as np
import cv2

class FaceEmbedder:
    def __init__(self):
        print("[INFO] Loading ArcFace embedding model...")
        self.model = insightface.app.FaceAnalysis(name="buffalo_l")
        self.model.prepare(ctx_id=0)   # CPU mode
        print("[INFO] Model loaded.")

    def get_embedding(self, img_path):
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face and get embedding
        faces = self.model.get(img)

        if len(faces) == 0:
            raise ValueError(f"No face found in image: {img_path}")

        # ArcFace returns a 512-dimensional embedding
        return faces[0].embedding

    def compare(self, img1, img2, threshold=0.8):
        emb1 = self.get_embedding(img1)
        emb2 = self.get_embedding(img2)

        # Cosine similarity
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        is_match = sim > threshold

        return float(sim), is_match