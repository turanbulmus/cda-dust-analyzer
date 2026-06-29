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
    print("Starting batch inference on inference dataset...")
    
    # 1. Load data
    inf_data_path = "cda_dust_agent/data/raw/cda_inf.parquet"
    if not os.path.exists(inf_data_path):
        print(f"Error: {inf_data_path} does not exist.")
        return
        
    print(f"Loading data from {inf_data_path}")
    df = pd.read_parquet(inf_data_path)
    
    # 2. Sample 1000 rows
    print("Sampling 1000 rows...")
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)
    else:
        df_sample = df
        print(f"Warning: Dataset only has {len(df)} rows, using all.")
        
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
    
    # 9. Parse and save to CSV
    print("Parsing results to CSV...")
    from cda_dust_agent.tools.utils import parse_response
    
    preds = []
    with open(out_file, 'r') as f:
        for line in f:
            preds.append(json.loads(line))
            
    parsed_data = []
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
            
        pred_id = parsed.get("id")
        pred_class = parsed.get("class") or parsed.get("class_label") or "Noise"
        explanation = parsed.get("explanation", "")
        
        parsed_data.append({
            "sclk": pred_id,
            "predicted_class": pred_class,
            "explanation": explanation
        })
        
    results_df = pd.DataFrame(parsed_data)
    results_csv = "cda_dust_agent/data/results/inf_results.csv"
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    print(f"Saved results to {results_csv}")

if __name__ == "__main__":
    main()
