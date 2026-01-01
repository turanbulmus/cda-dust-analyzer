from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()

import argparse
import sys

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} to {destination_file_name}")
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Batch Prediction Results")
    parser.add_argument("--uri", type=str, help="Full GCS URI of the predictions.jsonl file", default=None)
    args = parser.parse_args()

    # Priority: 1. CLI Argument, 2. .env variable, 3. Fail
    uri = args.uri or os.environ.get("BATCH_OUTPUT_URI")

    if not uri or "your-bucket-name" in uri or "YYYY-MM-DD" in uri:
        print("Error: No valid GCS URI provided.")
        print("Usage: python download_results.py --uri gs://bucket/path/predictions.jsonl")
        print("OR set BATCH_OUTPUT_URI in your .env file.")
        sys.exit(1)

    print(f"Downloading from: {uri}")
    bucket_name = uri.replace("gs://", "").split("/")[0]
    blob_name = "/".join(uri.replace("gs://", "").split("/")[1:])
    
    download_blob(bucket_name, blob_name, "predictions.jsonl")
