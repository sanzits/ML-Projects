import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FOLDER = "data"            # Root directory of face folders
IMG_SIZE = (100, 100)            # CNN input size (height, width)
EMBED_DIM = 128                  # Size of final embedding vector
EPOCHS = 10
BATCH_SIZE = 8


# ============================================================
# 1. Load images from folder structure
# ============================================================
def load_dataset():
    """
    Loads images from directory structure:
        faces/person1/*.png
        faces/person2/*.jpg
    Returns:
        dataset = {
            "person1": [img_array1, img_array2, ...],
            "person2": [img_array1, img_array2, ...]
        }
    Each image is scaled to [0,1].
    """
    dataset = {}

    for person in os.listdir(DATA_FOLDER):
        person_path = os.path.join(DATA_FOLDER, person)

        # Skip hidden files like .DS_Store
        if person.startswith(".") or not os.path.isdir(person_path):
            continue

        dataset[person] = []
        for img_file in os.listdir(person_path):
            if img_file.startswith("."):
                continue

            img_path = os.path.join(person_path, img_file)

            try:
                # Load image -> resize -> convert to array -> normalize
                img = load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
                img = img_to_array(img) / 255.0
                dataset[person].append(img)

            except Exception as e:
                print(f"Skipping invalid file: {img_path} ({e})")

    return dataset


# ============================================================
# 2. Create positive and negative image pairs
# ============================================================
def create_pairs(dataset, pairs_per_person=5):
    """
    Creates training pairs for Siamese network.

    Output:
        X1: first image in pair
        X2: second image in pair
        y : label (1 = same person, 0 = different people)
    """

    X1, X2, y = [], [], []
    persons = list(dataset.keys())

    for person in persons:
        images = dataset[person]

        # ---- Positive pairs (same person) ----
        for _ in range(pairs_per_person):
            if len(images) < 2:
                continue
            img1, img2 = random.sample(images, 2)
            X1.append(img1)
            X2.append(img2)
            y.append(1)

        # ---- Negative pairs (different people) ----
        for _ in range(pairs_per_person):
            img1 = random.choice(images)

            # pick random different person
            neg_person = random.choice([p for p in persons if p != person])
            img2 = random.choice(dataset[neg_person])

            X1.append(img1)
            X2.append(img2)
            y.append(0)

    return np.array(X1), np.array(X2), np.array(y)


# ============================================================
# 3. Build the embedding model (CNN)
# ============================================================
def build_embedding_model():
    """
    CNN that converts a face image into a vector (embedding).
    This embedding captures the "identity" of the face.
    """
    inp = Input(shape=(*IMG_SIZE, 3))

    x = Conv2D(32, (3,3), activation="relu")(inp)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3,3), activation="relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3,3), activation="relu")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(EMBED_DIM, activation="relu")(x)

    return Model(inp, x, name="EmbeddingNetwork")


# ============================================================
# 4. Build Siamese network using shared embedding model
# ============================================================
def build_siamese_model(embedding_model):
    """
    Siamese Network:
    - Takes 2 images
    - Generates 2 embeddings (same CNN shared)
    - Computes absolute difference
    - Dense layer outputs probability of "same person"
    """
    input_a = Input(shape=(*IMG_SIZE, 3))
    input_b = Input(shape=(*IMG_SIZE, 3))

    emb_a = embedding_model(input_a)
    emb_b = embedding_model(input_b)

    # Absolute difference between embeddings
    diff = Lambda(lambda t: tf.abs(t[0] - t[1]))([emb_a, emb_b])

    # Final prediction: 1 = same person, 0 = different
    out = Dense(1, activation="sigmoid")(diff)

    siamese = Model([input_a, input_b], out, name="SiameseNetwork")

    siamese.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return siamese


# ============================================================
# 5. MAIN TRAINING EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n[INFO] Loading dataset...")
    dataset = load_dataset()
    print(f"[INFO] People found: {list(dataset.keys())}")

    print("[INFO] Creating image pairs...")
    X1, X2, y = create_pairs(dataset)
    print(f"[INFO] Total pairs created: {len(y)}")

    print("[INFO] Building embedding model...")
    embedding_model = build_embedding_model()
    embedding_model.summary()

    print("[INFO] Building Siamese network...")
    siamese_model = build_siamese_model(embedding_model)
    siamese_model.summary()

    print("[INFO] Training...")
    siamese_model.fit(
        [X1, X2], y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True
    )

    print("\n[INFO] Training complete! Saving embedding model...")
    embedding_model.save("face_embedding_model.keras")
    print("[INFO] Saved as face_embedding_model.keras\n")