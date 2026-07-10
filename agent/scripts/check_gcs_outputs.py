import sys
from google.cloud import storage

def main():
    client = storage.Client(project="turan-genai-bb")
    bucket = client.bucket("turansgenaibb")
    
    print("Listing files in gs://turansgenaibb/input/ (VLM request chunks):")
    blobs = list(bucket.list_blobs(prefix="input/batch_requests_part"))
    for b in blobs[:10]:
        print(f" - {b.name} (Size: {b.size} bytes, Created: {b.time_created})")
    print(f"Total files: {len(blobs)}")
    
    print("\nListing files in gs://turansgenaibb/output/bypassed/ (Bypassed Noise):")
    b_blobs = list(bucket.list_blobs(prefix="output/bypassed/"))
    for b in b_blobs[:10]:
        print(f" - {b.name} (Size: {b.size} bytes, Created: {b.time_created})")
    print(f"Total files: {len(b_blobs)}")

if __name__ == "__main__":
    main()
