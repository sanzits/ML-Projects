from face_embedder import FaceEmbedder
import json

embedder = FaceEmbedder()

# Add your known people here:
known_people = {
    "Sanchit": "known_faces/Sanchit.png",
    "Anupriya": "known_faces/Anupriya.png"
}

db = {}

for name, img_path in known_people.items():
    emb_list = embedder.extract_face_embeddings(img_path)

    if len(emb_list) == 0:
        print(f"[WARNING] No face found for {name}")
        continue

    # Take the biggest face or the first one
    db[name] = emb_list[0].tolist()

with open("face_db.json", "w") as f:
    json.dump(db, f, indent=4)

print("[INFO] Face database created!")