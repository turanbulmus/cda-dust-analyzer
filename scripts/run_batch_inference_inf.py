"""
This script runs a specific large-scale batch inference job on a sample of 1000 rows
from the training dataset (excluding prompt study samples).

This differs from the standard agent run which typically evaluates on the test dataset
defined in the config.
"""
import os
import json
import time
import subprocess
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path to find cda_dust_agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from cda_dust_agent.config import Config
from cda_dust_agent.tools.utils import create_batch_input_file
from google.cloud import storage

configs = Config()
configs.agent_settings.inference_path = "batch" # Force batch mode

def main():
    print("Starting batch inference on training dataset sample...")
    
    # 1. Load data
    train_data_path = "cda_dust_agent/data/raw/cda_train.parquet"
    sampled_data_path = "cda_dust_agent/data/prompt_optimizer/sampled_train.parquet"
    
    if not os.path.exists(train_data_path):
        print(f"Error: {train_data_path} does not exist.")
        return
        
    print(f"Loading data from {train_data_path}")
    df = pd.read_parquet(train_data_path)
    
    # Exclude samples used in prompt study
    if os.path.exists(sampled_data_path):
        print(f"Loading excluded samples from {sampled_data_path}")
        sampled_df = pd.read_parquet(sampled_data_path)
        excluded_sclks = set(sampled_df['sclk'].dropna().astype(str))
        print(f"Excluding {len(excluded_sclks)} samples used in prompt study.")
        # Ensure sclk is string for matching
        df['sclk_str'] = df['sclk'].astype(str)
        df_filtered = df[~df['sclk_str'].isin(excluded_sclks)].drop(columns=['sclk_str'])
        print(f"Rows remaining after exclusion: {len(df_filtered)}")
    else:
        print("No sampled_train.parquet found, proceeding with all training data.")
        df_filtered = df
        
    # 2. Sample 1000 rows
    print("Sampling 1000 rows...")
    if len(df_filtered) > 1000:
        df_sample = df_filtered.sample(n=1000, random_state=42).reset_index(drop=True)
    else:
        df_sample = df_filtered
        print(f"Warning: Dataset only has {len(df_filtered)} rows, using all.")
        
    # 3. Load few-shot examples from cache if available
    few_shot_examples = []
    cache_file = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
    if os.path.exists(cache_file):
        print(f"Loading few-shot examples from {cache_file}")
        import base64
        with open(cache_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if "image_base64" in entry:
                        entry["image"] = base64.b64decode(entry["image_base64"])
                    few_shot_examples.append(entry)
        print(f"Loaded {len(few_shot_examples)} examples.")
    else:
        print("No few-shot cache found. Running without few-shot examples or you might want to run the agent first to generate them.")
        
    # 4. Generate JSONL file
    print("Generating JSONL file for batch inference...")
    jsonl_file, _ = create_batch_input_file(df_sample, few_shot_examples=few_shot_examples, limit=None)
    print(f"Generated {jsonl_file}")
    
    # 5. Upload to GCS
    project_id = configs.agent_settings.project_id
    bucket_name = configs.agent_settings.bucket_name
    
    print(f"Uploading {jsonl_file} to GCS bucket: {bucket_name}")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
    
    blob = bucket.blob(f"input/{jsonl_file}")
    blob.chunk_size = 10 * 1024 * 1024
    blob.upload_from_filename(jsonl_file, timeout=600)
    gcs_source = f"gs://{bucket.name}/input/{jsonl_file}"
    print(f"Uploaded to {gcs_source}")
    
    # 6. Submit Batch Job
    access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
    job_display_name = f"cda-inf-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    global_batch_req = {
        "displayName": job_display_name,
        "model": f"publishers/google/models/{configs.agent_settings.model}",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [gcs_source]}
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": f"gs://{bucket.name}/output"}
        }
    }
    
    req_file = "batch_request_inf.json"
    with open(req_file, "w") as f:
        json.dump(global_batch_req, f)
        
    curl_command = [
        "curl", "-s", "-X", "POST",
        f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/batchPredictionJobs",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json; charset=utf-8",
        "-d", f"@{req_file}"
    ]
    
    print("Submitting Batch Job via curl...")
    result = subprocess.run(curl_command, capture_output=True, text=True)
    
    if os.path.exists(req_file):
        os.remove(req_file)
        
    if result.returncode == 0 and "name" in result.stdout:
        response_json = json.loads(result.stdout)
        job_name = response_json.get("name")
        print(f"Job Submitted Successfully! Job Name: {job_name}")
    else:
        print(f"Error submitting job: {result.stderr}\nResponse: {result.stdout}")
        return
        
    # 7. Poll for completion
    check_url = f"https://aiplatform.googleapis.com/v1/{job_name}"
    print("Polling job status...")
    
    while True:
        # Refresh token in case it expires
        access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
        
        check_cmd = [
            "curl", "-s", "-X", "GET",
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
            print("Job Succeeded!")
            break
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
            print(f"Job Ended with state: {state}")
            if "error" in status_data:
                print(f"Error Details: {status_data['error']}")
            return
            
        time.sleep(30)
        
    # 8. Download results
    print("Downloading results...")
    blobs = list(bucket.list_blobs(prefix="output"))
    # Find files that match the job name or are recent
    prediction_blobs = [b for b in blobs if b.name.endswith(".jsonl") and "prediction" in b.name]
    
    if not prediction_blobs:
        print("Warning: No prediction JSONL files found in output directory.")
        return
        
    # Sort by time created to get the latest
    prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
    blob_to_download = prediction_blobs[0]
    
    output_dir = "cda_dust_agent/data/output"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "inf_predictions.jsonl")
    
    print(f"Downloading {blob_to_download.name} to {out_file}...")
    blob_to_download.download_to_filename(out_file)
    
    # 9. Parse and save to CSV with full analysis
    print("Parsing results and running analysis...")
    from cda_dust_agent.tools.utils import parse_response
    from sklearn.metrics import classification_report, confusion_matrix
    
    preds = []
    with open(out_file, 'r') as f:
        for line in f:
            preds.append(json.loads(line))
            
    # Create a truth map from the sampled data
    truth_map = {}
    for k, v in df_sample.set_index('sclk')['class'].to_dict().items():
        try:
            clean_k = str(int(float(k)))
        except ValueError:
            clean_k = str(k)
        truth_map[clean_k] = str(v)
        
    y_true = []
    y_pred = []
    explanations = []
    matched_sclks = []
    
    for p in preds:
        resp_text = ""
        try:
            if 'response' in p:
                resp_text = p['response']['candidates'][0]['content']['parts'][0]['text']
            elif 'predictions' in p:
                resp_text = p['predictions'][0]['candidates'][0]['content']['parts'][0]['text']
        except Exception:
            pass
            
        parsed = parse_response(resp_text)
        
        if isinstance(parsed, list):
            parsed = parsed[0] if len(parsed) > 0 else {}
        elif not isinstance(parsed, dict):
            parsed = {}
            
        raw_label = parsed.get("class") or parsed.get("class_label") or "Noise"
        raw_label_str = str(raw_label).strip().lower()
        
        # Dynamic matching based on unique classes in truth_map
        unique_classes_lower = {str(k).lower(): str(k) for k in truth_map.values()}
        pred_label = "Noise"
        for cls_lower, cls_real in unique_classes_lower.items():
            if cls_lower in raw_label_str:
                pred_label = cls_real
                break
                
        pred_id = parsed.get("id")
        explanation = parsed.get("explanation", "")
        
        if pred_id is not None:
            try:
                clean_id = str(int(float(pred_id)))
            except ValueError:
                clean_id = str(pred_id)
                
            if clean_id in truth_map:
                matched_sclks.append(clean_id)
                y_true.append(truth_map[clean_id])
                y_pred.append(pred_label)
                explanations.append(explanation)
                
    print(f"Parsed {len(y_true)} matched predictions.")
    
    # Save to CSV (overwriting the same file as requested)
    results_df = pd.DataFrame({
        "sclk": matched_sclks,
        "true_class": y_true,
        "predicted_class": y_pred,
        "explanation": explanations
    })
    results_csv = "cda_dust_agent/data/results/inf_results.csv"
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    print(f"Saved results to {results_csv}")
    
    # Run evaluation
    target_names = sorted(list({str(v) for v in truth_map.values()}))
    if not target_names:
        target_names = ['4', '1', 'Noise']
        
    report = classification_report(y_true, y_pred, labels=target_names)
    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    
    print("\n--- Evaluation Results ---")
    print(report)
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(f"Labels: {target_names}")
    print(cm)

if __name__ == "__main__":
    main()
