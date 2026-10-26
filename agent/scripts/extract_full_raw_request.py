import json
from google.cloud import storage

# Configure paths
PROJECT_ID = "turan-genai-bb"
BUCKET_NAME = "turansgenaibb"
INPUT_PREFIX = "input/batch_requests_part-00000-of-00059.jsonl"
ARTIFACTS_DIR = "/usr/local/google/home/turanbulmus/.gemini/jetski/brain/6caed9ec-1116-4f47-9bcb-aaac12b8740a"

def main():
    print("Connecting to GCS bucket...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    blob = bucket.blob(INPUT_PREFIX)
    print(f"Reading first line of input file: {blob.name}...")
    
    # Read the first line of the file in GCS
    first_line = None
    with blob.open("r") as f:
        for line in f:
            if line.strip():
                first_line = line.strip()
                break
                
    if not first_line:
        print("Error: Input file is empty.")
        return
        
    output_path = os.path.join(ARTIFACTS_DIR, "full_raw_batch_request.json")
    print(f"Writing complete, untruncated JSON request to {output_path}...")
    
    # Write the raw line directly to output file (preserving all base64 data and JSON structure)
    with open(output_path, "w") as f_out:
        f_out.write(first_line)
        
    print("Full raw request successfully saved!")

if __name__ == "__main__":
    import os
    main()
