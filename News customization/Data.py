import requests
import pandas as pd
from google.cloud import storage
import os

API_KEY = "bc0d8809b4c64f15bf768a88e38658a7"
BASE_URL = "https://newsapi.org/v2/everything"

def get_news(query, language="en", page_size=20):
    """
    Fetch news articles using NewsAPI.
    
    Args:
        query (str): Search keyword (e.g., "stock market").
        language (str): Language code (default: English).
        page_size (int): Number of results per page (max 100).
    
    Returns:
        pandas.DataFrame with news results.
    """
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}, {response.text}")

    data = response.json()
    articles = data.get("articles", [])

    # Convert to DataFrame
    df = pd.DataFrame(articles)
    if not df.empty:
        df = df[["source", "author", "title", "description", "url", "publishedAt", "content"]]
        # Expand source.name into its own column
        df["source"] = df["source"].apply(lambda x: x.get("name") if isinstance(x, dict) else x)
    
    return df

def upload_to_gcs(local_file, bucket_name, blob_name):
    """Uploads a file to Google Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file)
    print(f"✅ Uploaded {local_file} to gs://{bucket_name}/{blob_name}")


# 🔍 Example usage
if __name__ == "__main__":
    df = get_news("artificial intelligence", page_size=10)
    print(df.head())

    # Save to CSV
    df.to_csv("news_ai.csv", index=False)
    print("Saved news_ai.csv")
    upload_to_gcs("news_ai.csv", "sanchit-news-data", "news_ai.csv")