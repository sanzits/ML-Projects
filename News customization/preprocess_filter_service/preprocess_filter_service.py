import pandas as pd
import json
from google.cloud import storage
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import io

# ---------- Flask App ----------
app = Flask(__name__)

# ---------- Load Local Model ----------
model = SentenceTransformer("model/")

# ---------- Topics of Interest ----------
USER_INTERESTS = ["finance", "economy", "stock market", "geo-politics", "technology"]
interest_embeddings = model.encode(USER_INTERESTS, convert_to_tensor=True)

# ---------- Email Config ----------
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

# ---------- Helper Functions ----------

def download_from_gcs(bucket_name, source_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"File {source_blob_name} not found in bucket {bucket_name}")
    content = blob.download_as_text()
    return json.loads(content)

def upload_to_gcs(bucket_name, destination_blob_name, df):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

def send_email(subject, html_content):
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        return True
    except Exception as e:
        print(f"Email Error: {e}")
        return False

# ---------- API Endpoints ----------

@app.route("/process", methods=["POST"])
def preprocess_and_filter():
    data = request.get_json()
    bucket_name = data["bucket"]
    file_name = data["file"]

    try:
        articles = download_from_gcs(bucket_name, file_name)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    if not articles:
        return jsonify({"message": "No articles found"}), 200

    df = pd.DataFrame(articles)

    if "content" not in df.columns:
        return jsonify({"error": "No 'content' field in articles"}), 400

    # Clean content
    df["content"] = df["content"].fillna("").astype(str)
    df = df[df["content"].str.strip() != ""]

    if df.empty:
        return jsonify({"message": "No valid content to process"}), 200

    # Batch encode
    embeddings = model.encode(df["content"].tolist(), convert_to_tensor=True)
    relevance_scores = util.cos_sim(embeddings, interest_embeddings).max(dim=1).values
    df["relevance"] = relevance_scores.cpu().numpy()

    # Filter relevant articles
    filtered_df = df[df["relevance"] > 0.4].copy()

    if filtered_df.empty:
        return jsonify({"message": "No relevant articles found"}), 200

    # Upload to GCS
    output_path = file_name.replace("news_", "filtered_news_").replace(".json", ".csv")
    upload_to_gcs(bucket_name, output_path, filtered_df)

    return jsonify({"message": f"✅ Uploaded to {output_path}", "count": len(filtered_df)})

@app.route("/email", methods=["POST"])
def email_filtered_news():
    data = request.get_json()
    bucket_name = data["bucket"]
    file_name = data["file"]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    if not blob.exists():
        return jsonify({"error": f"File {file_name} not found in bucket"}), 404

    try:
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {e}"}), 500

    if df.empty:
        return jsonify({"error": "No data in file"}), 400

    html_table = df.to_html(index=False, escape=False)
    email_body = f"""
    <h2>📰 Filtered News Report</h2>
    <p>Source File: {file_name}</p>
    {html_table}
    <br><p>Regards,<br>Your Automated News System</p>
    """

    if send_email(subject="Filtered News Report", html_content=email_body):
        return jsonify({"message": "✅ Email sent successfully"})
    else:
        return jsonify({"error": "Failed to send email"}), 500

@app.route("/test", methods=["POST"])
def test_single_text():
    data = request.get_json()
    text = data.get("text", "")
    text_embedding = model.encode(text, convert_to_tensor=True)
    relevance = float(util.cos_sim(text_embedding, interest_embeddings).max())
    return jsonify({"text": text, "relevance": relevance})

@app.route("/test_email", methods=["GET"])
def test_email():
    html_content = """
    <h2>✅ Email Test from Preprocess Service</h2>
    <p>If you received this email, your Cloud Run email config works!</p>
    """
    if send_email("Cloud Run Email Test", html_content):
        return jsonify({"message": "✅ Test email sent successfully"})
    else:
        return jsonify({"error": "❌ Failed to send email"})

# ---------- Run Locally ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)