import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import numpy as np
import json

from face_embedder import FaceEmbedder


# ---------------------------
# EMAIL CONFIG
# ---------------------------
EMAIL_ADDRESS = "sanzit.s@gmail.com"
EMAIL_PASSWORD = "vrjv olge vzlz ssqp"

# Mapping person → email address
EMAIL_MAP = {
    "sanchit": "sanzit.s@gmail.com",
    "Anupriya":     "kaushikanupriya54@gmail.com"
}


# ---------------------------
# LOAD FACE DB & EMBEDDER
# ---------------------------
embedder = FaceEmbedder()

with open("face_db.json", "r") as f:
    face_db = json.load(f)

INPUT_FOLDER = "/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/Face_Detection_2/Photos"
VALID_EXT = (".jpg", ".jpeg", ".png")


# ---------------------------
# FACE MATCHING FUNCTION
# ---------------------------
def find_people_in_image(image_path, threshold=0.6):
    detected = []
    face_embeddings = embedder.extract_face_embeddings(image_path)

    for face_emb in face_embeddings:
        for name, known_emb in face_db.items():
            sim = embedder.cosine_similarity(face_emb, np.array(known_emb))
            if sim > threshold:
                detected.append((name, sim))

    return detected


# ---------------------------
# EMAIL SENDER FUNCTION
# ---------------------------
def send_email_with_photos(to_email, subject, body, attachment_paths):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    # attach files
    for file_path in attachment_paths:
        part = MIMEBase("application", "octet-stream")
        with open(file_path, "rb") as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename={os.path.basename(file_path)}")
        msg.attach(part)

    # send email
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    server.send_message(msg)
    server.quit()
    print(f"📧 Email sent to {to_email} with {len(attachment_paths)} photos")


# ---------------------------
# SCAN FOLDER & GROUP PHOTOS
# ---------------------------
photos_for_person = {name: [] for name in face_db.keys()}

print("🔍 Scanning folder...")

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(VALID_EXT):
        full_path = os.path.join(INPUT_FOLDER, filename)
        results = find_people_in_image(full_path)

        for person, sim in results:
            photos_for_person[person].append(full_path)
            print(f"{person} detected in {filename} (similarity={sim:.3f})")

# ---------------------------
# SEND EMAILS
# ---------------------------
print("\n📨 Sending emails...")

for person, photos in photos_for_person.items():
    if len(photos) == 0:
        continue

    if person not in EMAIL_MAP:
        print(f"⚠️ No email configured for {person}, skipping.")
        continue

    send_email_with_photos(
        to_email=EMAIL_MAP[person],
        subject="Your Photos",
        body=f"Hi {person},\n\nHere are the photos where you were detected.\n",
        attachment_paths=photos
    )