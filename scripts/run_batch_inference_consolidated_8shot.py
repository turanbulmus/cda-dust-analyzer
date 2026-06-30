"""
Runs a large-scale batch inference job on up to 2000 samples from the training dataset
using consolidated class labels (e.g., 3-Car -> 3) and exactly 8 examples per
consolidated class (i.e. 8 shots), ensuring no 'Noise' or '?' labels are sampled.
"""
import os
import json
import time
import base64
import subprocess
import sys
import asyncio
import logging
import pandas as pd
from datetime import datetime

# Add parent directory to path to find cda_dust_agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from cda_dust_agent.config import Config
from cda_dust_agent.tools.utils import create_batch_input_file, parse_response
from google.cloud import storage
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

configs = Config()
configs.agent_settings.inference_path = "batch" # Force batch mode

async def main():
    logger.info("Starting consolidated 8-shot batch inference on training dataset sample (excluding Noise/?)...")
    
    # 1. Load training dataset
    train_data_path = "cda_dust_agent/data/raw/cda_train.parquet"
    sampled_data_path = "cda_dust_agent/data/prompt_optimizer/sampled_train.parquet"
    
    if not os.path.exists(train_data_path):
        logger.error(f"Error: {train_data_path} does not exist.")
        return
        
    logger.info(f"Loading data from {train_data_path}")
    df = pd.read_parquet(train_data_path)
    
    # Consolidate labels in the training dataset (e.g. '3-Car' -> '3', '5-Na' -> '5')
    logger.info("Consolidating ground truth class labels...")
    df['class'] = df['class'].apply(lambda x: str(x).split('-')[0].strip() if pd.notna(x) else 'Noise')
    
    # Exclude samples used in prompt study
    if os.path.exists(sampled_data_path):
        logger.info(f"Loading excluded samples from {sampled_data_path}")
        sampled_df = pd.read_parquet(sampled_data_path)
        excluded_sclks = set(sampled_df['sclk'].dropna().astype(str))
        logger.info(f"Excluding {len(excluded_sclks)} samples used in prompt study.")
        df['sclk_str'] = df['sclk'].astype(str)
        df_filtered = df[~df['sclk_str'].isin(excluded_sclks)].drop(columns=['sclk_str'])
        logger.info(f"Rows remaining after exclusion: {len(df_filtered)}")
    else:
        logger.info("No sampled_train.parquet found, proceeding with all training data.")
        df_filtered = df
        
    # 2. Load cached examples and construct exactly 8 shots per consolidated class
    few_shot_examples = []
    cache_file = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
    
    consolidated_classes = ['Noise', '1', '2', '3', '4', '5', '?']
    class_example_counts = {c: 0 for c in consolidated_classes}
    
    if os.path.exists(cache_file):
        logger.info(f"Constructing 8-shot per consolidated class from {cache_file}")
        with open(cache_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    raw_lbl = entry.get("label", "Noise")
                    
                    # Map to consolidated label
                    if raw_lbl.startswith("Class "):
                        sub = raw_lbl[6:]
                        cons_sub = sub.split('-')[0].strip()
                        cons_key = cons_sub
                        cons_lbl = f"Class {cons_sub}"
                    else:
                        cons_key = "Noise"
                        cons_lbl = "Noise"
                        
                    if cons_key in class_example_counts and class_example_counts[cons_key] < 8:
                        if "image_base64" in entry:
                            entry["image"] = base64.b64decode(entry["image_base64"])
                        entry["label"] = cons_lbl
                        few_shot_examples.append(entry)
                        class_example_counts[cons_key] += 1
                        
        logger.info(f"Loaded {len(few_shot_examples)} consolidated few-shot examples (Distribution: {class_example_counts}).")
    else:
        logger.warning("No few-shot cache found. Please verify cached_examples.jsonl exists.")
        return
        
    used_sclk_ids = {ex['sclk'] for ex in few_shot_examples}
    
    # 3. Sample exactly 2000 rows (excluding few-shot reference SCLKs and 'Noise'/'?' labels)
    logger.info("Filtering out 'Noise' and '?' labels from evaluation sampling candidates as requested...")
    df_eval_candidates = df_filtered[
        (~df_filtered['sclk'].isin(used_sclk_ids)) & 
        (~df_filtered['class'].isin(['Noise', '?']))
    ]
    
    n_sample = min(2000, len(df_eval_candidates))
    logger.info(f"Sampling {n_sample} rows from the remaining {len(df_eval_candidates)} valid evaluation candidates...")
    
    if n_sample > 0:
        df_sample = df_eval_candidates.sample(n=n_sample, random_state=42).reset_index(drop=True)
    else:
        df_sample = df_eval_candidates
        
    # 4. Generate JSONL file for Batch Inference
    output_jsonl = "cda_dust_agent/data/input/batch_requests_consolidated_8shot.jsonl"
    logger.info(f"Generating JSONL input file: {output_jsonl}...")
    jsonl_file, _ = create_batch_input_file(df_sample, few_shot_examples=few_shot_examples, output_file=output_jsonl, limit=None)
    logger.info(f"Generated {jsonl_file}")
    
    # 5. Upload to GCS
    project_id = configs.agent_settings.project_id
    bucket_name = configs.agent_settings.bucket_name
    
    if not bucket_name or bucket_name == "your-gcs-bucket-name":
        bucket_name = os.getenv("GCS_BUCKET_NAME", "turan-genai-bb-cda-dust-analyzer")
    if not project_id or project_id == "your-google-cloud-project-id":
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "turan-genai-bb")
        
    logger.info(f"Uploading {jsonl_file} to bucket {bucket_name}...")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    gcs_blob_name = f"input/{os.path.basename(jsonl_file)}"
    blob = bucket.blob(gcs_blob_name)
    blob.upload_from_filename(jsonl_file)
    gcs_source = f"gs://{bucket_name}/{gcs_blob_name}"
    logger.info(f"Uploaded to {gcs_source}")
    
    # 6. Submit Batch Prediction Job via curl using gcloud access token
    access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
    if "ERROR" in access_token or not access_token:
        access_token = os.popen("gcloud auth print-access-token").read().strip()
        
    job_display_name = f"cda-inf-consolidated-8shot-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    global_batch_req = {
        "displayName": job_display_name,
        "model": f"publishers/google/models/{configs.agent_settings.model}",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [gcs_source]}
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": f"gs://{bucket.name}/output_8shot"}
        }
    }
    
    req_file = "batch_request_consolidated_8shot.json"
    with open(req_file, "w") as f:
        json.dump(global_batch_req, f)
        
    curl_command = [
        "curl", "-s", "-X", "POST",
        f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/batchPredictionJobs",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json; charset=utf-8",
        "-d", f"@{req_file}"
    ]
    
    logger.info("Submitting Batch Job via curl...")
    result = subprocess.run(curl_command, capture_output=True, text=True)
    
    if os.path.exists(req_file):
        os.remove(req_file)
        
    if result.returncode == 0 and "name" in result.stdout:
        response_json = json.loads(result.stdout)
        job_name = response_json.get("name")
        logger.info(f"Job Submitted Successfully! Job Name: {job_name}")
    else:
        logger.error(f"Error submitting job: {result.stderr}\nResponse: {result.stdout}")
        return
        
    # 7. Poll for Job Completion
    check_url = f"https://aiplatform.googleapis.com/v1/{job_name}"
    logger.info("Polling batch prediction job status (this typically takes several minutes)...")
    
    while True:
        access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
        if "ERROR" in access_token or not access_token:
            access_token = os.popen("gcloud auth print-access-token").read().strip()
            
        check_cmd = ["curl", "-s", "-X", "GET", check_url, "-H", f"Authorization: Bearer {access_token}"]
        check_res = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if check_res.returncode != 0:
            logger.error(f"Error checking status: {check_res.stderr}")
            await asyncio.sleep(30)
            continue
            
        status_data = json.loads(check_res.stdout)
        state = status_data.get("state", "UNKNOWN")
        logger.info(f"Job State: {state}")
        
        if state == "JOB_STATE_SUCCEEDED":
            break
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
            logger.error(f"Job Ended prematurely with state: {state}")
            return
            
        await asyncio.sleep(30)
        
    # 8. Download Results
    logger.info("Downloading prediction output files from GCS...")
    blobs = list(bucket.list_blobs(prefix="output_8shot"))
    prediction_blobs = [b for b in blobs if b.name.endswith(".jsonl") and "prediction" in b.name]
    
    if not prediction_blobs:
        logger.error("No prediction JSONL files found in destination.")
        return
        
    prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
    blob_to_download = prediction_blobs[0]
    
    out_file = "cda_dust_agent/data/output/consolidated_8shot_predictions.jsonl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    blob_to_download.download_to_filename(out_file)
    logger.info(f"Downloaded results to {out_file}")
    
    # 9. Parse and Evaluate
    logger.info("Parsing predictions and calculating evaluation metrics...")
    preds = []
    with open(out_file, 'r') as f:
        for line in f:
            preds.append(json.loads(line))
            
    truth_map = {str(row['sclk']): str(row['class']) for _, row in df_sample.iterrows()}
    
    y_true = []
    y_pred = []
    
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
        raw_label = parsed.get("class") or parsed.get("class_label") or "Noise"
        
        # Consolidate predicted label
        pred_label = str(raw_label).split('-')[0].strip()
        
        pred_id = parsed.get("id")
        if pred_id is not None:
            pred_id_str = str(pred_id)
            if pred_id_str in truth_map:
                y_true.append(truth_map[pred_id_str])
                y_pred.append(pred_label)
                
    logger.info(f"Matched {len(y_true)} predictions out of {len(df_sample)} submitted.")
    
    target_names = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=target_names)
    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    
    logger.info("\n--- Classification Report (Consolidated 8-Shot, Excl. Noise/?) ---\n" + report)
    logger.info(f"\n--- Confusion Matrix (Labels: {target_names}) ---\n{cm}")

if __name__ == "__main__":
    asyncio.run(main())
