import insightface
import numpy as np
import cv2


class FaceEmbedder:
    def __init__(self):
        print("[INFO] Loading ArcFace model...")
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)  # CPU mode
        print("[INFO] Model loaded.")

    def extract_face_embeddings(self, img_path):
        """
        Returns a list of embeddings for ALL faces in an image.
        Each embedding is a 512-dimensional vector.
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = self.app.get(img)

        embeddings = []
        for f in faces:
            embeddings.append(f.embedding)

        return embeddings

    def cosine_similarity(self, emb1, emb2):
        return float(
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        )