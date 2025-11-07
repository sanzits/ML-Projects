import requests
import json
from datetime import datetime
from google.cloud import storage
#import functions_framework  
import base64

API_KEY = "bc0d8809b4c64f15bf768a88e38658a7"
BUCKET_NAME = "sanchit-news-data"
QUERY = "finance OR economy OR stock market OR geo-politics OR technology"

PREPROCESS_URL = "https://preprocess-service-103861710535.us-central1.run.app/process"
EMAIL_URL = "https://preprocess-service-103861710535.us-central1.run.app/email"


def upload_to_gcs(bucket_name, blob_name, data):
    """Uploads JSON data to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")
    print(f"✅ Uploaded to gs://{bucket_name}/{blob_name}")


def trigger_endpoint(url, payload):
    """Safely POST to an external endpoint."""
    try:
        res = requests.post(url, json=payload, timeout=60)
        print(f"Triggered {url} → Status {res.status_code}: {res.text[:200]}")
        return res.status_code
    except Exception as e:
        print(f"Error calling {url}: {e}")
        return None


def run_fetch_news():
    """Fetches daily news → uploads → preprocess → email."""
    print("📡 Fetching news...")
    url = f"https://newsapi.org/v2/top-headlines?language=en&apiKey={API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])

    if not articles:
        print("⚠️ No news found.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_name = f"news_{timestamp}.json"
    upload_to_gcs(BUCKET_NAME, file_name, articles)

    preprocess_payload = {"bucket": BUCKET_NAME, "file": file_name}
    trigger_endpoint(PREPROCESS_URL, preprocess_payload)

    filtered_file = file_name.replace("news_", "filtered_news_").replace(".json", ".csv")
    email_payload = {"bucket": BUCKET_NAME, "file": filtered_file}
    trigger_endpoint(EMAIL_URL, email_payload)

    print("✅ News fetched, processed, and emailed.")


@functions_framework.cloud_event
def fetch_news_pubsub(cloud_event):
    """Triggered by Pub/Sub message."""
    print(f"🕓 Pub/Sub triggered at {datetime.utcnow().isoformat()}")

    try:
        # Optional: read Pub/Sub message data if you ever send parameters
        data = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
        print("Pub/Sub message:", data)
    except Exception:
        pass

    run_fetch_news()