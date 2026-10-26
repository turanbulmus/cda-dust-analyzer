import json
from google.cloud import storage

client = storage.Client(project='turan-genai-bb')
bucket = client.bucket('turansgenaibb')
blob = bucket.blob('output/prediction-model-2026-07-07T17:17:53.785156Z/predictions.jsonl')

content = blob.download_as_text()
lines = content.splitlines()
print(f"Total lines in file: {len(lines)}")

for i, line in enumerate(lines):
    if not line.strip():
        continue
    try:
        data = json.loads(line)
        print(f"\nLine {i+1} Keys: {list(data.keys())}")
        if "error" in data:
            print(f"Error on Line {i+1}: {data['error']}")
        elif "status" in data and data["status"] != "OK":
            print(f"Status on Line {i+1}: {data['status']}")
        else:
            # Let's inspect the prediction/response payload
            if "response" in data:
                resp = data["response"]
                print(f"Response Keys: {list(resp.keys())}")
                if "error" in resp:
                    print(f"Response Error: {resp['error']}")
            if "predictions" in data:
                preds = data["predictions"]
                print(f"Predictions Type: {type(preds)}")
    except Exception as e:
        print(f"Error parsing line {i+1}: {e}")
