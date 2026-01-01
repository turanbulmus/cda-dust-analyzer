"""
CDA Dust Analyzer Classification Pipeline
=========================================

This script manages the classification of Cosmic Dust Analyzer (CDA) spectra to identify
"Class 4" events using Google's Gemini 3.0 Pro model via Vertex AI.

Features:
- **Local Evaluation**: Rapidly test prompt logic and metrics on a subset of data using `google.genai` SDK.
- **Batch Processing**: Generate JSONL input files and submit large-scale jobs to Vertex AI Batch Prediction.
- **Automated Visualization**: Converts raw spectral data into high-contrast plots with visual guides (e.g., green target regions).
- **Structured Output**: Enforces consistent JSON output (boolean label + reasoning) for downstream analysis.

Usage:
    # 1. Run local evaluation (requires GOOGLE_CLOUD_API_KEY)
    python batch_classify.py --local-eval

    # 2. Run full batch processing (submit to Vertex AI)
    python batch_classify.py --bucket gs://YOUR_BUCKET_NAME

Dependencies:
    - google-genai
    - google-cloud-aiplatform
    - google-cloud-storage
    - pandas, matplotlib, pydantic
"""

import argparse
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

try:
    import vertexai
    # from vertexai.preview import generative_models # Legacy
    # from vertexai.preview.batch_prediction import BatchPredictionJob # Legacy for batch?
    from google import genai
    from google.genai import types
    from google.cloud import aiplatform, storage
    # Pydantic is already available if google.genai is installed
    from pydantic import BaseModel, Field
except ImportError:
    print("google-cloud-aiplatform not installed. Please install it.")
    exit(1)

# Load .env
load_dotenv()

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = "us-central1"
MODEL_ID = "gemini-3-pro-preview"
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-bucket-name")

# Centralized Prompts
SYSTEM_INSTRUCTION_TEXT = """You are an expert Cosmic Dust Classification System specializing in Time-of-Flight Mass Spectrometry data. Your sole objective is to correctly identify **Class 4** events while rejecting Class 1 and Noise distractors.

**INPUT DATA:**
- X-axis: Time-of-Flight (Indices). Note: Features may jitter/shift slightly (±20 indices).
- Y-axis: Amplitude (Signal Intensity).

**CLASS 4 DEFINITION (TARGET):**
To classify a sample as Class 4, it MUST exhibit a **Central Signal Complex**.
- **Location:** A distinct, high-amplitude feature located roughly between **X = 180 and X = 400** (specifically centered around 200-250 in nominal cases).
- **Shape:** This is not typically a single narrow line; look for a "jagged" shape, a double-peak, or a complex cluster of high activity rising **significantly** above the baseline.
- **Amplitude Requirement:** The signal in this region MUST be prominent. **Ignorable bumps, faint fluctuations, or low-amplitude noise in this region are NOT sufficient.**

**REJECTION CRITERIA (DISTRACTORS):**
1.  **The "Early Spike Only" (Class 1 / Simple Noise):** If the spectrum contains a prominent sharp spike at the far left (X < 50) but the region between X=180 and X=400 is flat (baseline noise only), REJECT.
2.  **The "Wall of Static" (Chaotic Noise):** If the signal consists of continuous, high-amplitude fluctuations across the entire X-axis without distinct, isolated signal peaks, REJECT.
3.  **Weak/Ambiguous Signals:** If the feature in the 180-400 range is weak, barely visible above baseline, or looks like random low-level noise, REJECT (Classify as Non-Target).

**OUTPUT:**
Provide a concise analysis of the 180-400 range and a final classification: "Class 4" or "Non-Target".
Return JSON: {"is_class_4": boolean, "reasoning": string}
"""

USER_PROMPT_TEXT = """Analyze the provided spectrum plot to determine if it belongs to **Class 4**.

Follow these steps:
1.  **Scan the Central Region (X = 180 to 400):** Is there a distinct, high-amplitude signal complex in this region? Describe its shape (jagged, double-peak, etc.).
2.  **Check for Rejection Features:** Is there *only* an early spike (X < 50) with a quiet central region? Is the signal just chaotic static? Is the central signal too weak/faint?
3.  **Conclusion:** Based on the above, is this Class 4?"""

def generate_spectrum_image_bytes(spectrum_data, title=None):
    """Generates a PNG byte buffer of the spectrum plot."""
    plt.figure(figsize=(12, 6))
    plt.plot(spectrum_data, color='black', linewidth=2)
    # Highlight Class 4 region
    plt.axvspan(180, 400, color='green', alpha=0.1, label='Class 4 Region')
    # Highlight Noise region
    plt.axvspan(0, 50, color='red', alpha=0.1, label='Noise Region')
    if title:
        plt.title(title)
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def get_few_shot_examples(df):
    """Extracts reference few-shot examples (Class 4, Class 1, Noise)."""
    examples = []
    
    # 1. Class 4 (Target)
    cls4_subset = df[df['class'].astype(str) == '4']
    if not cls4_subset.empty:
        # Use first one or specific high-quality one
        row = cls4_subset.iloc[0]
        img_bytes = generate_spectrum_image_bytes(row['spectrum'], title="Reference: Class 4")
        examples.append({
            "label": "Class 4",
            "image": img_bytes,
            "explanation": "POSITIVE MATCH. Note the distinct high-amplitude signal complex (jagged/double-peak) rising significantly above baseline in the Target Zone (X=180-400).",
            "sclk": row['sclk']
        })
    
    # 2. Class 1 (Distractor - Early Spike)
    cls1_subset = df[df['class'].astype(str) == '1']
    if not cls1_subset.empty:
        row = cls1_subset.iloc[0]
        img_bytes = generate_spectrum_image_bytes(row['spectrum'], title="Reference: Class 1 (Non-Target)")
        examples.append({
            "label": "Non-Target",
            "image": img_bytes,
            "explanation": "NEGATIVE MATCH. Contains a strong 'Early Spike' at X<50, but the Target Zone (X=180-400) is quiet/flat. This is Class 1 (Distractor), NOT Class 4.",
            "sclk": row['sclk']
        })
        
    # 3. Noise (Distractor - Static)
    noise_subset = df[df['class'] == 'Noise']
    if not noise_subset.empty:
        row = noise_subset.iloc[0]
        img_bytes = generate_spectrum_image_bytes(row['spectrum'], title="Reference: Noise (Non-Target)")
        examples.append({
            "label": "Non-Target",
            "image": img_bytes,
            "explanation": "NEGATIVE MATCH. Signal resembles random chaotic static or is too weak. No distinct isolated complex in the Target Zone.",
            "sclk": row['sclk']
        })
        
    return examples, [ex['sclk'] for ex in examples]

def create_batch_input_file(df, output_file='batch_requests.jsonl', limit=None):
    """Creates the JSONL input file for Gemini Batch with Few-Shot examples."""
    print(f"Generating batch input file: {output_file}...")
    
    # 1. Get Few-Shot Examples
    few_shot_examples, used_sclk_ids = get_few_shot_examples(df)
    print(f"Few-shot examples used (will be excluded from batch): {used_sclk_ids}")
    
    # We will prepend the few-shot examples to the USER prompt content effectively
    # by adding them as previous turns or just context parts.
    # Simplest for Gemini 3.0 is interleaved text/image parts.
    
    few_shot_parts = []
    if few_shot_examples:
        few_shot_parts.append({"text": "Here are reference examples:"})
        for ex in few_shot_examples:
            few_shot_parts.append({"text": f"Example: {ex['label']} ({ex['explanation']})"})
            # Need base64 for JSON file
            ex_b64 = base64.b64encode(ex['image']).decode('utf-8')
            few_shot_parts.append({"inline_data": {"mime_type": "image/png", "data": ex_b64}})
        few_shot_parts.append({"text": "Now, analyze the following spectrum:"})
    
    # Use centralized user prompt
    prompt_text = USER_PROMPT_TEXT

    requests = []
    
    rows = df.iterrows()
    if limit:
        rows = list(rows)[:limit]
    else:
        rows = list(rows)

    print(f"Processing {len(rows)} items for batch generation...")
    
    for idx, row in rows:
        # SKIP FEW-SHOT EXAMPLES to avoid leakage/redundancy
        if row['sclk'] in used_sclk_ids:
            continue
            
        spect = row['spectrum']
        # Convert to image bytes
        img_bytes = generate_spectrum_image_bytes(spect, title=f"Sample {row['sclk']}")
        # Encode for JSON
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Build contents
        # Structure: [Few-Shot Parts] + [Current Image] + [Prompt Text]
        current_parts = few_shot_parts.copy()
        current_parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        current_parts.append({"text": prompt_text})
        
        request = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": current_parts
                    }
                ],
                "systemInstruction": {
                    "parts": [{"text": SYSTEM_INSTRUCTION_TEXT}]
                },
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }
        }
        requests.append(json.dumps(request))
        
    with open(output_file, 'w') as f:
        for req in requests:
            f.write(req + '\n')
            
    print(f"Saved {len(requests)} requests to {output_file}")
    print(f"Saved {len(requests)} requests to {output_file}")
    return output_file, used_sclk_ids

def parse_response(response_text):
    """Parses JSON response from model."""
    if not response_text:
        return {"is_class_4": False, "reasoning": "Empty response text"}
    try:
        # cleanup markdown code blocks if present
        text = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Error parsing JSON: {e} | Text: {response_text}")
        return {"is_class_4": False, "reasoning": f"Parse Error: {response_text}"}

def run_local_evaluation(df, limit=20):
    """Runs local evaluation on a subset of data for immediate metrics."""
    print(f"Starting Local Evaluation with {MODEL_ID}...")
    
    # Initialize new SDK client
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
    
    if api_key:
        print("Using GOOGLE_CLOUD_API_KEY for authentication.")
        client = genai.Client(vertexai=True, api_key=api_key)
    else:
        print("No key found. Save it as an environment variable and try again.")
        return
     
    # Prepare Balanced Subset
    df_c4 = df[df['class'].astype(str) == '4']
    df_noise = df[df['class'] == 'Noise']
    
    n_samples = limit // 2
    subset = pd.concat([
        df_c4.head(n_samples),
        df_noise.head(n_samples)
    ]).sample(frac=1, random_state=42) # Shuffle
    
    print(f"Evaluating on {len(subset)} samples ({len(df_c4.head(n_samples))} Class 4, {len(df_noise.head(n_samples))} Noise)...")
    
    # Get Few-Shot Context (shared across calls)
    few_shot_context, _ = get_few_shot_examples(df)
    print(f"Loaded {len(few_shot_context)} few-shot examples.")

    # Use centralized system prompt
    # system_instruction_text is now SYSTEM_INSTRUCTION_TEXT global variable
    
    # Define Schema for Structured Output
    class ClassificationResult(BaseModel):
        is_class_4: bool = Field(description="True if the spectrum is Class 4 (has peak at ~640), False otherwise.")
        reasoning: str = Field(description="Detailed reasoning explaining why the spectrum matches or does not match Class 4 criteria.")

    # Build Config
    config = types.GenerateContentConfig(
        temperature=0.2, 
        top_p=0.95,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=ClassificationResult,
        system_instruction=[types.Part.from_text(text=SYSTEM_INSTRUCTION_TEXT)],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH"
        ),
    )

    results = []
    
    for i, row in subset.iterrows():
        s_id = row['sclk']
        true_label = str(row['class'])
        spectrum = row['spectrum']
        
        # Generate Image
        img_bytes = generate_spectrum_image_bytes(spectrum, title=f"SCLK: {s_id}")
        
        # Build Contents
        # Few-shot history + current user request
        contents = []
        
        # Add Few-shot (manually constructing history as Content objects)
        if few_shot_context:
            for item in few_shot_context:
                # User part
                contents.append(types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=f"Example: {item['label']} ({item['explanation']})"),
                        types.Part(inline_data=types.Blob(mime_type="image/png", data=item['image']))
                    ]
                ))
                # Model part
                # Construct expected output from item data
                expected_response = {
                    "is_class_4": item['label'] == "Class 4",
                    "reasoning": item['explanation']
                }
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=json.dumps(expected_response))]
                ))
        
        # Current Request
        current_parts = [
            types.Part.from_text(text=USER_PROMPT_TEXT),
            types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes))
        ]
        contents.append(types.Content(role="user", parts=current_parts))

        try:
            # Generate
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config
            )
            
            resp_text = response.text
            if not resp_text:
                 print(f"WARNING: No text in response for {s_id}. Candidate info: {response.candidates[0] if response.candidates else 'No candidates'}")
            
            resp_data = parse_response(resp_text)
            
            # Map prediction
            pred_label = "4" if resp_data.get("is_class_4") else "Noise"
            
            results.append({
                "sclk": s_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "reasoning": resp_data.get("reasoning", "No reasoning provided")
            })
            print(f"Sample {s_id}: True={true_label}, Pred={pred_label}")
            print(f"  Reasoning: {resp_data.get('reasoning', 'No reasoning provided')}")
            
        except Exception as e:
            print(f"Error calling API for {s_id}: {e}")
            results.append({
                "sclk": s_id,
                "true_label": true_label,
                "pred_label": "Error",
                "reasoning": str(e)
            })

    # ... (Metrics calculation remains same if I return results correctly) ...
    # Calculate Metrics
    y_true = [r['true_label'] for r in results if r['pred_label'] != 'Error']
    y_pred = [r['pred_label'] for r in results if r['pred_label'] != 'Error']
    
    if not y_true:
        print("No successful predictions.")
        return results

    # Normalize labels for metric calc (Target=4, Negative=Noise)
    y_true_bin = [1 if x == '4' else 0 for x in y_true]
    y_pred_bin = [1 if x == '4' else 0 for x in y_pred]
    
    # Simple Metrics
    tp = sum([1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1])
    fp = sum([1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1])
    fn = sum([1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0])
    tn = sum([1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 0])
    
    accuracy = (tp + tn) / len(y_true_bin) if y_true_bin else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision (Class 4): {precision:.2f}")
    print(f"Recall (Class 4): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    cm = [[tp, fn], [fp, tn]]
    print(f"Confusion Matrix (TP, FN, FP, TN):\n{cm}")
    
    return results

def run_pipeline(dry_run=False, bucket_uri=None):
    bucket_to_use = bucket_uri or f"gs://{BUCKET_NAME}"
    
    print(f"Initializing Vertex AI (Project: {PROJECT_ID})...")
    if not dry_run:
        aiplatform.init(project=PROJECT_ID, location=LOCATION, experiment='cda-dust-classification')
        
        # Create Experiment Run
        run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        aiplatform.start_run(run=run_name)
        aiplatform.log_params({
            "model": MODEL_ID,
            "prompt_version": "v2_few_shot",
            "bucket": bucket_to_use
        })
        print(f"Started Experiment Run: {run_name}")

    # Load Data
    try:
        df = pd.read_parquet('data/cda_sample.parquet')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if dry_run:
        print("[DRY-RUN] Processing subset of 3 items...")
        df_subset = df.head(3)
        jsonl_file, used_ids = create_batch_input_file(df_subset, limit=3)
        print(f"[DRY-RUN] JSONL created at {jsonl_file}. Ready for upload to {bucket_to_use}.")
        return

    # Full Run
    jsonl_file, used_ids = create_batch_input_file(df)
    
    # Upload to GCS
    from google.cloud import storage
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME) # Use configured bucket name variable
    if bucket_uri:
         # simple parsing if URI provided
         bucket_name_arg = bucket_uri.replace("gs://", "").split("/")[0]
         bucket = storage_client.bucket(bucket_name_arg)
    
    blob = bucket.blob(f"input/{jsonl_file}")
    blob.upload_from_filename(jsonl_file)
    gcs_source = f"gs://{bucket.name}/input/{jsonl_file}"
    print(f"Uploaded {jsonl_file} to {gcs_source}")

    # BATCH SUBMISSION VIA CURL (Global Endpoint)
    # The user requested to use the global endpoint via direct API call to bypass SDK regional restrictions.
    
    print(f"\nSubmitting Batch Job via curl to GLOBAL endpoint...")

    # Use ADC token as recommended to avoid CBA/CLI token warnings
    access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
    if not access_token:
        print("Error: Could not retrieve gcloud access token. Ensure you are logged in via `gcloud auth login`.")
        return

    job_display_name = f"cda-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"    
    global_batch_req = {
        "displayName": job_display_name,
        "model": f"publishers/google/models/{MODEL_ID}",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [gcs_source]}
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": f"gs://{bucket.name}/output"}
        }
    }
    
    # --- EXPERIMENT TRACKING STARTS HERE ---
    try:
        experiment_name = "cda-dust-analyzer-experiment"
        run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        print(f"Initializing Experiment Run: {run_name} inside {experiment_name}...")
        aiplatform.init(project=PROJECT_ID, location=LOCATION, experiment=experiment_name)
        
        with aiplatform.start_run(run_name) as run:
            # 1. Log Hyperparameters
            params = {
                "model_id": MODEL_ID,
                "temperature": 0.2, # Configured in batch payload
                "top_p": 0.95,
                "max_output_tokens": 8192,
                "prompt_file": "batch_classify.py (Internal)",
                "input_file": gcs_source,
                "job_display_name": job_display_name,
                "few_shot_ids": str(used_ids)
            }
            run.log_params(params)
            
            # 3. Create Artifact for Prompt (System + User + Few-Shot)
            few_shot_ex, _ = get_few_shot_examples(df)
            few_shot_desc = "\n".join([f"- {ex['label']}: {ex['explanation']} (SCLK: {ex['sclk']})" for ex in few_shot_ex])
            
            prompt_data = {
                "system_instruction": SYSTEM_INSTRUCTION_TEXT,
                "user_prompt": USER_PROMPT_TEXT,
                "few_shot_examples": few_shot_desc
            }
            
            # Save Locally
            with open("experiment_prompts.json", "w") as f:
                json.dump(prompt_data, f, indent=2)
            print("Saved experiment_prompts.json locally.")

            try:
                print("Creating Prompt Artifact in Vertex Metadata...")
                prompt_artifact = aiplatform.Artifact.create(
                    schema_title="system.Artifact",
                    display_name="cda_prompt_v2_enhanced",
                    metadata=prompt_data
                )
                print(f"Prompt Artifact created: {prompt_artifact.resource_name}")
            except Exception as e:
                print(f"Warning: Failed to create prompt artifact: {e}")
            
            # 2. Log Metrics (Placeholder - will be updated by analyze_results.py)
            run.log_metrics({"status": "submitted"})
            
            # Save Run Name to file for analyze_results.py to pick up later
            with open("last_run_config.json", "w") as f:
                json.dump({"run_name": run_name, "experiment_name": experiment_name}, f)
                
            print(f"Experiment tracking initialized.")
            print(f"View Experiment Run Here: https://console.cloud.google.com/vertex-ai/experiments/experiments/{experiment_name}/runs/{run_name}?project={PROJECT_ID}")
        
    except Exception as e:
        print(f"Warning: Failed to log experiment: {e}")
    # --- EXPERIMENT TRACKING ENDS HERE ---

    # Save request to file for curl
    with open("batch_request.json", "w") as f:
        json.dump(global_batch_req, f)
        
    curl_command = [
        "curl", "-X", "POST",
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/batchPredictionJobs",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json; charset=utf-8",
        "-d", "@batch_request.json"
    ]
    
    import subprocess
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True)
        print("Response Code:", result.returncode)
        print("Response Body:", result.stdout)
        
        if result.returncode == 0 and "name" in result.stdout:
            response_json = json.loads(result.stdout)
            job_name = response_json.get("name")
            print(f"Batch Job Submitted Successfully! Job Name: {job_name}")
            
            # --- POLLING LOOP ---
            print("\nWaiting for Batch Job to complete...")
            print("This may take a while depending on the number of items.")
            
            job_id = job_name.split("/")[-1]
            check_url = f"https://aiplatform.googleapis.com/v1/{job_name}"
            
            while True:
                # Poll status
                check_cmd = [
                    "curl", "-X", "GET",
                    check_url,
                    "-H", f"Authorization: Bearer {access_token}"
                ]
                check_res = subprocess.run(check_cmd, capture_output=True, text=True)
                
                if check_res.returncode != 0:
                    print(f"Error checking status: {check_res.stderr}")
                    time.sleep(30)
                    continue
                    
                status_data = json.loads(check_res.stdout)
                state = status_data.get("state", "UNKNOWN")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Job State: {state}")
                
                if state == "JOB_STATE_SUCCEEDED":
                    print("\nJob Completed Successfully!")
                    break
                elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
                    print(f"\nJob Ended with state: {state}")
                    if "error" in status_data:
                        print(f"Error Details: {status_data['error']}")
                    return
                
                time.sleep(30)
                
            # --- DOWNLOAD RESULTS ---
            print("\nDownloading results...")
            try:
                # Output URI is directory, we need to find the file(s)
                # Convention: output/prediction-model-...
                # But we can just list the output directory in GCS
                
                output_prefix = f"output"
                # predictions.jsonl might be split or named differently, usually predictions.jsonl for direct simple runs
                # But actually Vertex Batch Output is usually a directory containing *.jsonl files
                
                # Let's use the explicit bucket download logic
                from google.cloud import storage
                storage_client = storage.Client(project=PROJECT_ID)
                bucket = storage_client.bucket(bucket.name) # Re-use bucket obj
                
                blobs = list(bucket.list_blobs(prefix=output_prefix))
                # Find the one that ends with .jsonl and is not empty
                prediction_blobs = [b for b in blobs if b.name.endswith(".jsonl") and "prediction" in b.name]
                
                if not prediction_blobs:
                    print("Warning: No prediction JSONL files found in output directory.")
                else:
                    # Sort by creation time (newest first)
                    prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
                    blob_to_download = prediction_blobs[0]
                    
                    print(f"Downloading {blob_to_download.name} (created: {blob_to_download.time_created}) to predictions.jsonl...")
                    blob_to_download.download_to_filename("predictions.jsonl")
                    print("Download complete.")
                    
                    # --- RUN ANALYSIS ---
                    print("\nStarting Analysis...")
                    import analyze_results
                    analyze_results.main()
                    
            except Exception as e:
                print(f"Error during download/analysis: {e}")
                
        else:
            print(f"Error submitting batch job via curl: {result.stderr}")
            print(f"Response: {result.stdout}")
            
    except Exception as e:
        print(f"Exception during curl execution: {e}")
    finally:
        if os.path.exists("batch_request.json"):
            os.remove("batch_request.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run locally without API calls')
    parser.add_argument('--bucket', type=str, help='GCS Bucket URI for Batch API')
    parser.add_argument('--local-eval', action='store_true', help='Run immediate local evaluation on subset')
    args = parser.parse_args()
    
    if args.local_eval:
        df = pd.read_parquet('data/cda_sample.parquet')
        run_local_evaluation(df)
    else:
        run_pipeline(dry_run=args.dry_run, bucket_uri=args.bucket)
