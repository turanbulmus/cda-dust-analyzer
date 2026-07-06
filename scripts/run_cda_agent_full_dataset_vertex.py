import os
import sys
import json
import time
import subprocess
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from google.cloud import storage, aiplatform

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from cda_dust_agent.config import Config
from cda_dust_agent.tools.utils import create_batch_input_file, parse_response
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

configs = Config()
# Override config parameters as requested by user:
configs.agent_settings.prompt_optimization = False   # Skip prompt optimizer leg
configs.agent_settings.test_mode = False             # Full dataset execution (no sampling limit)
configs.agent_settings.inference_path = "batch"      # Force Vertex AI Batch Inference

PROJECT_ID = configs.agent_settings.project_id
BUCKET_NAME = configs.agent_settings.bucket_name.replace("gs://", "")
LOCATION = configs.agent_settings.location
MODEL_ID = configs.agent_settings.model

def main():
    print("=" * 80)
    print("CDA DUST ANALYZER AGENT: FULL DATASET VERTEX AI EXPERIMENT RUN")
    print(f"Project ID: {PROJECT_ID} | Bucket: {BUCKET_NAME} | Model: {MODEL_ID}")
    print("Prompt Optimization: SKIPPED (Using complemented prompts.py)")
    print("Dataset Mode: FULL DATASET (test_mode=False)")
    print("=" * 80)
    
    # 1. Load Data
    train_path = "cda_dust_agent/data/raw/cda_train.parquet"
    test_path = "cda_dust_agent/data/testing/cda_test.parquet"
    
    if os.path.exists(test_path):
        print(f"Loading test dataset from {test_path}...")
        df = pd.read_parquet(test_path)
    elif os.path.exists(train_path):
        print(f"Loading train dataset from {train_path}...")
        df = pd.read_parquet(train_path)
    else:
        print("Dataset parquet file not found. Fetching from HuggingFace...")
        import huggingface_hub
        REPO_ID = "CosmicDustGroup/cassini-cda-spectra"
        FILENAME = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"
        file_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
        df = pd.read_parquet(file_path)

    print(f"Total spectra loaded for full run: {len(df)}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    
    # 2. Few-Shot Examples (Loaded from cache if available)
    few_shot_examples = []
    cache_file = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
    if os.path.exists(cache_file):
        print(f"Loading cached few-shot examples from {cache_file}...")
        import base64
        with open(cache_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if "image_base64" in entry:
                        entry["image"] = base64.b64decode(entry["image_base64"])
                    few_shot_examples.append(entry)
        print(f"Loaded {len(few_shot_examples)} few-shot examples.")

    # 3. Create Batch Input JSONL (Full dataset, limit=None)
    print("\nGenerating batch prediction JSONL for full dataset...")
    jsonl_filename = f"cda_batch_input_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    jsonl_file, used_ids = create_batch_input_file(
        df, 
        few_shot_examples=few_shot_examples, 
        limit=None, 
        output_file=f"cda_dust_agent/data/input/{jsonl_filename}"
    )
    print(f"Batch JSONL generated: {jsonl_file}")

    # 4. Upload to GCS Bucket
    print(f"\nUploading {jsonl_file} to GCS (gs://{BUCKET_NAME}/input/)...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    blob_name = f"input/{os.path.basename(jsonl_file)}"
    blob = bucket.blob(blob_name)
    blob.chunk_size = 10 * 1024 * 1024  # 10MB chunking
    blob.upload_from_filename(jsonl_file, timeout=600)
    gcs_source = f"gs://{BUCKET_NAME}/{blob_name}"
    print(f"Uploaded successfully to {gcs_source}")

    # 5. Submit Vertex AI Batch Prediction Job
    job_display_name = f"cda-full-dataset-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"\nSubmitting Vertex AI Batch Prediction Job '{job_display_name}'...")
    
    aiplatform.init(project=PROJECT_ID, location="global")
    model_to_use = f"publishers/google/models/{MODEL_ID}"
    
    try:
        job = aiplatform.BatchPredictionJob.create(
            job_display_name=job_display_name,
            model_name=model_to_use,
            instances_format="jsonl",
            gcs_source=gcs_source,
            predictions_format="jsonl",
            gcs_destination_prefix=f"gs://{BUCKET_NAME}/output",
        )
        print(f"Job Submitted Successfully! Job Resource Name: {job.name}")
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e):
            fallback_model = "publishers/google/models/gemini-2.5-flash"
            print(f"Model {model_to_use} not found (404), falling back to {fallback_model}...")
            job = aiplatform.BatchPredictionJob.create(
                job_display_name=job_display_name,
                model_name=fallback_model,
                instances_format="jsonl",
                gcs_source=gcs_source,
                predictions_format="jsonl",
                gcs_destination_prefix=f"gs://{BUCKET_NAME}/output",
            )
            print(f"Fallback Job Submitted Successfully! Job Resource Name: {job.name}")
        else:
            raise e

    # 6. Poll Batch Job Status
    print("\nPolling job completion status...")
    while not job.has_ended:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Job State: {job.state.name}")
        time.sleep(30)
        job.refresh()
        
    print(f"Job completed with state: {job.state.name}")
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Error: Job did not succeed. State: {job.state.name}")
        return

    # 7. Download Results & Evaluate
    print("\nDownloading prediction results from GCS...")
    blobs = list(bucket.list_blobs(prefix="output"))
    prediction_blobs = [b for b in blobs if b.name.endswith(".jsonl") and "prediction" in b.name]
    
    if not prediction_blobs:
        print("Error: No prediction JSONL files found in GCS output directory.")
        return
        
    prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
    latest_blob = prediction_blobs[0]
    
    out_file = "cda_dust_agent/data/output/full_dataset_predictions.jsonl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    latest_blob.download_to_filename(out_file)
    print(f"Downloaded results to {out_file}")

    # Parse predictions
    preds = []
    with open(out_file, 'r') as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    truth_map = {str(row['sclk']): str(row['class']) for _, row in df.iterrows()}
    y_true, y_pred = [], []

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
        pred_label = str(parsed.get("class") or parsed.get("class_label") or "Noise").strip()
        pred_id = parsed.get("id")
        if pred_id is not None and str(pred_id) in truth_map:
            y_true.append(truth_map[str(pred_id)])
            y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nMatched Predictions: {len(y_true)} / {len(df)}")
    print(f"Overall Classification Accuracy: {acc:.2%}")
    
    target_names = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\nClassification Report:\n", report)

    # Save results CSV
    res_df = pd.DataFrame({"true_class": y_true, "predicted_class": y_pred})
    results_csv = "cda_dust_agent/data/results/results.csv"
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    res_df.to_csv(results_csv, index=False)

    # 8. Log Experiment Parameters & Metrics to Vertex AI Experiments
    experiment_name = "cda-dust-analyzer-experiment"
    run_name = f"full-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print(f"\nLogging run to Vertex AI Experiments ({experiment_name} -> {run_name})...")
    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        experiment=experiment_name
    )
    
    params = {
        "model": MODEL_ID,
        "inference_path": "batch",
        "prompt_optimization": False,
        "test_mode": False,
        "total_samples": len(df),
        "few_shot_count": len(few_shot_examples),
        "job_id": job.name
    }
    
    metrics = {
        "accuracy": float(acc),
        "total_matched": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    }
    
    # Add per-class recall/precision metrics
    for cls in set(y_true):
        cls_y_true = [1 if t == cls else 0 for t in y_true]
        cls_y_pred = [1 if p == cls else 0 for p in y_pred]
        metrics[f"recall_{cls}"] = float(recall_score(cls_y_true, cls_y_pred, zero_division=0))
        metrics[f"precision_{cls}"] = float(precision_score(cls_y_true, cls_y_pred, zero_division=0))

    with aiplatform.start_run(run=run_name):
        aiplatform.log_params(params)
        aiplatform.log_metrics(metrics)

    print("\nVertex AI Experiment Logging Complete!")
    print(f"Experiment Name: {experiment_name}")
    print(f"Run Name: {run_name}")
    print("=" * 80)

if __name__ == "__main__":
    main()
