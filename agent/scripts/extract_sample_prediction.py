import os
import json
import base64
from google.cloud import storage

# Configure paths
PROJECT_ID = "turan-genai-bb"
BUCKET_NAME = "turansgenaibb"
RAW_PRED_PREFIX = "output/prediction-model-2026-07-07T19:05:24.605584Z"
ARTIFACTS_DIR = "/usr/local/google/home/turanbulmus/.gemini/jetski/brain/6caed9ec-1116-4f47-9bcb-aaac12b8740a"

def main():
    print("Connecting to GCS bucket...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    # 1. Find the first JSONL prediction shard
    blobs = list(bucket.list_blobs(prefix=RAW_PRED_PREFIX))
    pred_blobs = [b for b in blobs if b.name.endswith(".jsonl")]
    if not pred_blobs:
        print("No prediction JSONL shards found in GCS prefix.")
        return
        
    target_blob = sorted(pred_blobs, key=lambda x: x.name)[0]
    print(f"Reading first line of shard: {target_blob.name}...")
    
    # Read GCS file line-by-line using streaming to avoid loading massive files
    line_data = None
    with target_blob.open("r") as f:
        for line in f:
            if line.strip():
                line_data = json.loads(line)
                break
                
    if not line_data:
        print("Empty prediction file.")
        return
        
    print("Record loaded successfully. Extracting components...")
    
    # 2. Extract System Instruction and Prompts
    request = line_data.get("request", {})
    response = line_data.get("response", {})
    
    system_instruction = request.get("systemInstruction", {}).get("parts", [{}])[0].get("text", "N/A")
    contents_parts = request.get("contents", [{}])[0].get("parts", [])
    
    # Get user instructions and few-shots
    user_prompt = contents_parts[0].get("text", "N/A")
    target_details = contents_parts[-1].get("text", "N/A")
    
    # Find the target image (which is the last inline_data element in parts)
    target_img_b64 = None
    few_shot_images = []
    
    for part in contents_parts:
        if "inline_data" in part and part["inline_data"] is not None:
            img_data = part["inline_data"].get("data")
            if img_data:
                few_shot_images.append(img_data)
            
    # The last image in the contents is the target spectrum being classified
    if few_shot_images:
        target_img_b64 = few_shot_images[-1]
        # Remove it from few-shot list
        few_shot_images = few_shot_images[:-1]
        
    # 3. Decode and save the target spectrum image to artifacts images/
    os.makedirs(os.path.join(ARTIFACTS_DIR, "images"), exist_ok=True)
    target_img_path = os.path.join(ARTIFACTS_DIR, "images", "target_spectrum.png")
    
    if target_img_b64:
        print(f"Decoding and saving target spectrum image to {target_img_path}...")
        img_bytes = base64.b64decode(target_img_b64)
        with open(target_img_path, "wb") as f_img:
            f_img.write(img_bytes)
    else:
        print("Warning: Target image not found in request.")

    # Save a couple of few-shot example images to artifacts too to show them
    for i, fs_img_b64 in enumerate(few_shot_images[:2]): # save first two few-shots
        fs_path = os.path.join(ARTIFACTS_DIR, "images", f"fewshot_example_{i+1}.png")
        img_bytes = base64.b64decode(fs_img_b64)
        with open(fs_path, "wb") as f_img:
            f_img.write(img_bytes)
            
    # 4. Extract VLM output
    vlm_output_raw = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
    vlm_parsed = {}
    try:
        vlm_parsed = json.loads(vlm_output_raw.replace("```json", "").replace("```", "").strip())
    except Exception:
        pass
        
    # 5. Format and write the markdown file
    md_path = os.path.join(ARTIFACTS_DIR, "sample_prediction_walkthrough.md")
    print(f"Generating walkthrough markdown at {md_path}...")
    
    markdown_content = f"""# VLM Sample API Prediction Walkthrough

This document showcases a single end-to-end API request and response record extracted from the full Cassini CDA dataset run on **Gemini 3.5 Flash**.

---

## 1. System Instructions (Prompt Role)

The VLM was initialized with the following system instructions:

```text
{system_instruction}
```

---

## 2. User Request Payload

### A. Classification Guidelines
```text
{user_prompt}
```

### B. Few-Shot Context Examples (Sample Images)
Each request embeds **16 few-shot examples** to guide the classification. Here are two of the loaded visual examples:

| Example 1 | Example 2 |
| :---: | :---: |
| ![Few-Shot 1](images/fewshot_example_1.png) | ![Few-Shot 2](images/fewshot_example_2.png) |

### C. Target Spectrum Under Analysis
Below is the plot of the spectrum submitted for classification:

![Target Spectrum to Classify](images/target_spectrum.png)

*   **Request SCLK metadata**: `{target_details}`

---

## 3. VLM Response (Raw Output JSON)

The API returned the following response:

```json
{vlm_output_raw}
```

### Parsed Attributes:
*   **Target SCLK ID**: `{vlm_parsed.get('id', 'N/A')}`
*   **Predicted Class**: `Class {vlm_parsed.get('class', 'Noise')}`
*   **VLM Explanation**: *"{vlm_parsed.get('explanation', 'N/A')}"*
"""

    with open(md_path, "w") as f_md:
        f_md.write(markdown_content)
        
    print("Markdown walkthrough successfully created!")

if __name__ == "__main__":
    main()
