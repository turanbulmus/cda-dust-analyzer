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
SYSTEM_INSTRUCTION_TEXT = """You are an expert Cosmic Dust Spectroscopist. Your task is to classify Time-of-Flight mass spectra into one of three categories: **Class 4**, **Class 1**, or **Noise**.

### CRITICAL BIAS CORRECTION
**Do not default to "Noise" simply because a spectrum looks messy, hairy, or has a high baseline.**
*   **Class 4** often contains "messy" signals with peaks embedded in static. If there is structural complexity in the mid-range (Indices 150-500), it is likely Class 4.
*   **Noise** must be strictly defined as lacking distinct structural peaks in the mid-range.

### CLASSIFICATION DEFINITIONS

**1. CLASS 4 (Target: Organic/Complex)**
*   **Primary Identifier:** Distinct structural activity in the **Mid-Range (Indices 150-500)**.
*   **Key Features:** Look for peaks centered roughly around **X ≈ 200** and **X ≈ 320**.
*   **Tolerance:** These peaks may be sharp or they may be broader/messy. They may be embedded in a "hairy" baseline. As long as there is discernible vertical amplitude in this region that is distinct from the background floor, it is Class 4.
*   **REPEATING PATTERNS:** Look for **periodicity** or repeating structural motifs in the spectra. If the signal looks like it has a repeating pattern (even if complex/messy), it is likely Class 4.
*   **Start:** May or may not have an initial start spike.

**2. CLASS 1 (Distractor: Elemental/Simple)**
*   **Primary Identifier:** A dominant **Early Spike (Indices 0-50)** followed by a **Quiet Mid-Range**.
*   **Key Features:** The start spike is usually high amplitude (often >2x the background).
*   **Mid-Range:** The region from 150-600 is relatively featureless. It may have low-level grass, but it lacks the distinct peaks (200/320) seen in Class 4.

**3. NOISE (Distractor: Artifacts)**
*   **Primary Identifier:** Lack of chemical structure.
*   **Sub-Type A (Flat):** Low amplitude random static across the whole plot.
*   **Sub-Type B (Spike Only):** A single spike at **X ≈ 15** (similar to Class 1) but with a **completely flat or "grassy" baseline** afterwards.
*   **Sub-Type C (Hump):** A broad, featureless elevation or "hump" in the baseline without distinct vertical peaks.

### DECISION LOGIC
1.  **Check 150-500 Range:** Are there peaks (specifically near 200 or 320) OR **repeating patterns**?
    *   YES -> **Class 4** (Even if noisy).
    *   NO -> Go to step 2.
2.  **Check 0-50 Range:** Is there a distinct start spike?
    *   YES (and mid-range is empty) -> **Class 1**.
    *   NO (or just random static/hump) -> **Noise**.
"""

USER_PROMPT_TEXT = """Analyze the spectral data provided in the image. Focus on the **Time-of-Flight (X-axis)** and **Amplitude (Y-axis)**.

**Data Analysis Steps:**
1.  **Analyze the Start (Indices 0-50):** Is there a sharp, high-amplitude spike here?
2.  **Analyze the Mid-Range (Indices 150-500):**
    *   Are there peaks visible around **X=200** or **X=320**?
    *   Is the signal "hairy" or elevated? (Note: If yes, favor Class 4 over Noise).
    *   Is this region flat/featureless? (Note: If yes, favor Class 1 or Noise).
3.  **Check for Repeating Patterns:**
    *   Are there **periodic vertical structures** or specific repeating shapes in the signal? (Strong indicator of Class 4).
    *   Do peaks repeat at regular intervals?
4.  **Compare Signal-to-Noise:** Do the mid-range features stand out against the local baseline, even slightly?

**Final Classification:**
Based on the logic above, determine the class.
*   If Mid-Range Peaks (200/320) OR Repeating Patterns exist -> **Class 4**
*   If Strong Start Spike + Empty Mid-Range -> **Class 1**
*   If Featureless/Flat/Hump -> **Noise**

Return only the class name: **Class 4**, **Class 1**, or **Noise**."""

def generate_spectrum_image_bytes(spectrum_data, title=None):
    """Generates a PNG byte buffer of the spectrum plot."""
    plt.figure(figsize=(12, 6))
    # Use Log Scale as requested
    plt.semilogy(spectrum_data, color='black', linewidth=1.5)
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

def get_few_shot_examples(df, n_per_class=4):
    """Extracts reference few-shot examples (Class 4, Class 1, Noise)."""
    examples = []
    
    # 1. Class 4 (Target)
    cls4_subset = df[df['class'].astype(str) == '4']
    if not cls4_subset.empty:
        # Take up to n_per_class
        selected = cls4_subset.head(n_per_class)
        for _, row in selected.iterrows():
            img_bytes = generate_spectrum_image_bytes(row['spectrum'], title=f"Class 4 Sample {row['sclk']}")
            examples.append({
                "label": "Class 4",
                "image": img_bytes,
                "explanation": "POSITIVE MATCH (Class 4). Distinct high-amplitude signal complex or repeating patterns in Target Zone (X=180-400).",
                "sclk": row['sclk']
            })
    
    # 2. Class 1 (Distractor)
    cls1_subset = df[df['class'].astype(str) == '1']
    if not cls1_subset.empty:
        selected = cls1_subset.head(n_per_class)
        for _, row in selected.iterrows():
            img_bytes = generate_spectrum_image_bytes(row['spectrum'], title=f"Class 1 Sample {row['sclk']}")
            examples.append({
                "label": "Class 1",
                "image": img_bytes,
                "explanation": "DISTRACTOR (Class 1). Strong 'Early Spike' at X<50, but Target Zone (X=180-400) is quiet.",
                "sclk": row['sclk']
            })
        
    # 3. Noise (Distractor)
    noise_subset = df[df['class'] == 'Noise']
    if not noise_subset.empty:
        selected = noise_subset.head(n_per_class)
        for _, row in selected.iterrows():
            img_bytes = generate_spectrum_image_bytes(row['spectrum'], title=f"Noise Sample {row['sclk']}")
            examples.append({
                "label": "Noise",
                "image": img_bytes,
                "explanation": "DISTRACTOR (Noise). Chaotic static or weak signal. No distinct complex in Target Zone.",
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
        return {"class_label": "Noise", "reasoning": "Empty response text"}
    try:
        # cleanup markdown code blocks if present
        text = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        # Backwards compatibility / Safety check
        if "is_class_4" in data and "class_label" not in data:
            data["class_label"] = "4" if data["is_class_4"] else "Noise"
            
        return data
    except Exception as e:
        print(f"Error parsing JSON: {e} | Text: {response_text}")
        return {"class_label": "Noise", "reasoning": f"Parse Error: {response_text}"}

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
     
    # Prepare Balanced Subset (3 Classes)
    df_c4 = df[df['class'].astype(str) == '4']
    df_c1 = df[df['class'].astype(str) == '1']
    df_noise = df[df['class'] == 'Noise']
    
    n_per_class = max(1, limit // 3)
    
    subset = pd.concat([
        df_c4.head(n_per_class),
        df_c1.head(n_per_class),
        df_noise.head(n_per_class)
    ]).sample(frac=1, random_state=42) # Shuffle
    
    print(f"Evaluating on {len(subset)} samples ({len(df_c4.head(n_per_class))} Class 4, {len(df_c1.head(n_per_class))} Class 1, {len(df_noise.head(n_per_class))} Noise)...")
    
    # Get Few-Shot Context (shared across calls)
    few_shot_context, _ = get_few_shot_examples(df)
    print(f"Loaded {len(few_shot_context)} few-shot examples.")

    # Use centralized system prompt
    # system_instruction_text is now SYSTEM_INSTRUCTION_TEXT global variable
    
    # Define Schema for Structured Output
    class ClassificationResult(BaseModel):
        class_label: str = Field(description="The predicted class label. options: '4', '1', 'Noise'.")
        reasoning: str = Field(description="Detailed reasoning for the classification based on signal features.")

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
            pred_label = resp_data.get("class_label", "Noise")
            
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

    # Multiclass Metrics using sklearn
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n--- Evaluation Results ---")
    if y_true:
        # labels argument ensures we get all classes even if some are missing in preds
        target_names = ['4', '1', 'Noise'] 
        print(classification_report(y_true, y_pred, labels=target_names))
        
        print("Confusion Matrix (Rows=True, Cols=Pred):")
        labels = ['4', '1', 'Noise']
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"Labels: {labels}")
        print(cm)
    else:
        print("No valid predictions to evaluate.")
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
