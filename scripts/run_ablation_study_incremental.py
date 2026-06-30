import os
import io
import json
import base64
import asyncio
import time
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from google import genai
from google.genai import types
from google.cloud import storage
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("google_genai").setLevel(logging.WARNING)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cda_dust_agent.config import Config
from cda_dust_agent.prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT, ANNOTATION_USER_PROMPT
from cda_dust_agent.tools.utils import generate_spectrum_image_bytes, parse_response

# 1. Explanation Generation (Online concurrent, default DPI)
async def generate_explanation(client, model_id, spectrum_array, label, sclk, semaphore, references=None):
    async with semaphore:
        # Standardize labels
        def format_class_label(cls):
            cls_str = str(cls)
            if cls_str.lower().startswith("class"):
                return cls_str
            return f"Class {cls_str}" if cls_str.lower() != 'noise' else "Noise"

        target_label = format_class_label(label)
        img_bytes = generate_spectrum_image_bytes(np.array(spectrum_array), title=f"{target_label} Sample {sclk}")
        
        if references:
            from cda_dust_agent.prompts import CONTRASTIVE_ANNOTATION_USER_PROMPT, SYSTEM_INSTRUCTION_TEXT
            
            ref_descriptions = ""
            for idx, (other_cls, _) in enumerate(references):
                char_code = chr(66 + idx) # B, C, D, E...
                ref_descriptions += f"- Image {char_code}: Reference Spectrum of '{format_class_label(other_cls)}'\n"
                
            prompt = CONTRASTIVE_ANNOTATION_USER_PROMPT.format(
                label=target_label,
                reference_descriptions=ref_descriptions.strip()
            )
            
            parts = [
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=img_bytes, mime_type="image/png") # Image A
            ]
            for _, other_img_bytes in references:
                parts.append(types.Part.from_bytes(data=other_img_bytes, mime_type="image/png"))
                
            contents = [types.Content(role="user", parts=parts)]
        else:
            from cda_dust_agent.prompts import ANNOTATION_USER_PROMPT, SYSTEM_INSTRUCTION_TEXT
            prompt = ANNOTATION_USER_PROMPT.format(label=label)
            contents = [
                types.Content(role="user", parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                ])
            ]
        
        retries = 3
        for attempt in range(retries):
            try:
                response = await client.aio.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION_TEXT
                    )
                )
                return {
                    "label": label,
                    "sclk": sclk,
                    "image": img_bytes,
                    "explanation": response.text.strip(),
                    "image_base64": base64.b64encode(img_bytes).decode('utf-8')
                }
            except Exception as e:
                print(f"Failed to generate explanation for {label} (sclk: {sclk}): {e}. Retrying {attempt+1}/{retries}...")
                await asyncio.sleep(2 ** attempt)
        return None

def preprocess_huggingface_data(dataset_type="both"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_L_path = os.path.join(base_dir, "cda_dust_agent/data/raw/cda_train_L.parquet")
    train_H_path = os.path.join(base_dir, "cda_dust_agent/data/raw/cda_train_H.parquet")
    
    dfs = []
    if dataset_type in ["L", "both"] and os.path.exists(train_L_path):
        dfs.append(pd.read_parquet(train_L_path))
    if dataset_type in ["H", "both"] and os.path.exists(train_H_path):
        dfs.append(pd.read_parquet(train_H_path))
        
    if not dfs:
        raise FileNotFoundError(f"Processed data files for '{dataset_type}' not found.")
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df[full_df['class'] != '?'].copy()
    return full_df

# 2. Get or Create Reproducible Training SCLKs
def get_or_create_train_sclks(full_df, output_path="cda_dust_agent/data/ablation_study/train_sclks.json"):
    max_train_map = {
        '5': 16,
        '5-Na': 16
    }
    existing_sclks = {}
    if os.path.exists(output_path):
        print(f"Loading existing training SCLKs from {output_path}...")
        with open(output_path, 'r') as f:
            try:
                existing_sclks = json.load(f)
            except Exception as e:
                print(f"Error loading {output_path}: {e}. Will regenerate.")
                existing_sclks = {}
        
    target_max = 24
    train_sclks = {}
    updated = False
    for cls, group in full_df.groupby('class'):
        cls_str = str(cls)
        n_target = min(target_max, max_train_map.get(cls_str, target_max), len(group))
        existing_list = existing_sclks.get(cls_str, [])
        
        if len(existing_list) >= n_target:
            train_sclks[cls_str] = existing_list[:n_target]
        else:
            updated = True
            print(f"Class {cls_str} has {len(existing_list)} existing SCLKs, extending to {n_target}...")
            remaining_group = group[~group['sclk'].isin(existing_list)]
            n_more = n_target - len(existing_list)
            if n_more > 0 and len(remaining_group) > 0:
                n_more = min(n_more, len(remaining_group))
                sampled_more = remaining_group.sample(n=n_more, random_state=123)['sclk'].tolist()
                new_list = existing_list + sampled_more
            else:
                new_list = existing_list
            train_sclks[cls_str] = new_list
            
    if updated or not existing_sclks:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(train_sclks, f, indent=4)
        print(f"Saved updated training SCLKs to {output_path}")
    return train_sclks

# 3. Generate VLM Explanations Pool for Training SCLKs
async def generate_explanations_pool(client, model_id, full_df, train_sclks):
    all_train_sclks = []
    for cls, sclks in train_sclks.items():
        all_train_sclks.extend(sclks)
        
    train_df = full_df[full_df['sclk'].isin(all_train_sclks)].copy()
    
    # Pre-render and cache all training spectrum images once
    print("Pre-rendering and caching training spectrum images...")
    from cda_dust_agent.tools.utils import generate_spectrum_image_bytes
    import random
    
    image_cache = {}
    for _, row in train_df.iterrows():
        sclk = row['sclk']
        cls = row['class']
        label = f"Class {cls}" if str(cls).lower() != 'noise' else "Noise"
        img_bytes = generate_spectrum_image_bytes(np.array(row['spectrum']), title=f"{label} Reference Sample {sclk}")
        image_cache[sclk] = img_bytes
        
    api_semaphore = asyncio.Semaphore(10)
    explanation_tasks = []
    
    # Set seed for reproducible dynamic sampling
    random.seed(123)
    
    print(f"Generating explanations for {len(train_df)} training samples concurrently with dynamic contrastive references...")
    for _, row in train_df.iterrows():
        target_sclk = row['sclk']
        target_cls = row['class']
        
        # Select one random reference SCLK from each of the other classes
        references = []
        for other_cls, other_sclks in train_sclks.items():
            if other_cls != target_cls:
                # Pick a random SCLK from the other class's SCLK list
                other_sclk = random.choice(other_sclks)
                references.append((other_cls, image_cache[other_sclk]))
                
        task = generate_explanation(client, model_id, row['spectrum'], row['class'], row['sclk'], api_semaphore, references=references)
        explanation_tasks.append(task)
        
    results = await asyncio.gather(*explanation_tasks)
    explanations_pool = [r for r in results if r is not None]
    print(f"Successfully generated {len(explanations_pool)} reference explanations.")
    return explanations_pool

# 4. Generate k-shot requests JSONL file
def create_k_shot_batch_input_file(test_df, explanations_pool, k, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    class_explanations = {}
    for ex in explanations_pool:
        cls = ex['label']
        if cls not in class_explanations:
            class_explanations[cls] = []
        class_explanations[cls].append(ex)
        
    max_train_map = {
        '5': 16,
        '5-Na': 16
    }
    
    k_shot_examples = []
    for cls, exs in class_explanations.items():
        k_actual = min(k, max_train_map.get(cls, k))
        k_shot_examples.extend(exs[:k_actual])
        
    used_sclk_ids = [ex['sclk'] for ex in k_shot_examples]
    
    few_shot_parts = [{"text": CLASSIFICATION_USER_PROMPT}]
    if k_shot_examples:
        few_shot_parts.append({"text": "Here are reference examples:"})
        for ex in k_shot_examples:
            few_shot_parts.append({"text": f"Example: {ex['label']} ({ex['explanation']})"})
            few_shot_parts.append({"inline_data": {"mime_type": "image/png", "data": ex['image_base64']}})
        few_shot_parts.append({"text": "Now, analyze the following spectrum:"})
        
    requests = []
    for _, row in test_df.iterrows():
        if row['sclk'] in used_sclk_ids:
            continue
            
        # Note: Do not set DPI, let generate_spectrum_image_bytes use default
        img_bytes = generate_spectrum_image_bytes(row['spectrum'], title=f"Sample {row['sclk']}")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        current_parts = few_shot_parts.copy()
        current_parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        current_parts.append({"text": f"Sample ID (sclk): {row['sclk']} | k: {k}"})
        
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
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "STRING", "description": "The ID of the run (sclk)"},
                            "class": {"type": "STRING", "description": "The predicted class label"},
                            "explanation": {"type": "STRING", "description": "Explanation for the prediction"},
                            "k": {"type": "INTEGER", "description": "The value of k shots used"}
                        },
                        "required": ["id", "class", "explanation", "k"]
                    }
                }
            }
        }
        requests.append(json.dumps(request))
        
    with open(output_file, 'w') as f:
        for req in requests:
            f.write(req + '\n')
            
    print(f"Generated JSONL file with {len(requests)} requests at {output_file}")
    return output_file

# 5. Upload to GCS (with tqdm progress bar) and submit Batch Job
def submit_and_poll_batch_job(configs, local_jsonl_path, k):
    project_id = configs.agent_settings.project_id
    bucket_name = configs.agent_settings.bucket_name
    
    filename = os.path.basename(local_jsonl_path)
    print(f"Uploading {local_jsonl_path} to GCS bucket: {bucket_name}")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
    
    gcs_input_path = f"input/{filename}"
    blob = bucket.blob(gcs_input_path)
    
    import tqdm
    class ProgressFileWrapper(object):
        def __init__(self, fileobj, total_size):
            self.fileobj = fileobj
            self.total_size = total_size
            self.pbar = tqdm.tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=f"Uploading GCS (k={k})",
                leave=True
            )
            self.bytes_read = 0

        def read(self, size=-1):
            chunk = self.fileobj.read(size)
            if chunk:
                self.bytes_read += len(chunk)
                self.pbar.update(len(chunk))
            return chunk

        def seek(self, offset, whence=0):
            self.fileobj.seek(offset, whence)
            current_pos = self.fileobj.tell()
            self.pbar.n = current_pos
            self.pbar.refresh()

        def tell(self):
            return self.fileobj.tell()

        def close(self):
            self.pbar.close()
            self.fileobj.close()

    total_size = os.path.getsize(local_jsonl_path)
    with open(local_jsonl_path, 'rb') as f:
        wrapped_file = ProgressFileWrapper(f, total_size)
        # Set chunk size to 10MB (must be a multiple of 256KB) and increase timeout to 10 minutes
        blob.chunk_size = 10 * 1024 * 1024
        blob.upload_from_file(wrapped_file, content_type="application/json", timeout=600)
        
    gcs_source = f"gs://{bucket.name}/{gcs_input_path}"
    print(f"Uploaded to {gcs_source}")
    
    access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
    job_display_name = f"cda-ablation-k{k}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
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
    
    req_file = f"batch_request_k{k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(req_file, "w") as f:
        json.dump(global_batch_req, f)
        
    curl_command = [
        "curl", "-s", "-X", "POST",
        f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/batchPredictionJobs",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json; charset=utf-8",
        "-d", f"@{req_file}"
    ]
    
    print(f"Submitting Batch Job for k={k} to Vertex AI...")
    result = subprocess.run(curl_command, capture_output=True, text=True)
    
    if os.path.exists(req_file):
        os.remove(req_file)
        
    if result.returncode == 0 and "name" in result.stdout:
        response_json = json.loads(result.stdout)
        job_name = response_json.get("name")
        print(f"Job Submitted! Name: {job_name}")
    else:
        print(f"Error submitting job: {result.stderr}\nResponse: {result.stdout}")
        raise RuntimeError("Failed to submit Batch Prediction Job.")
        
    check_url = f"https://aiplatform.googleapis.com/v1/{job_name}"
    print("Polling job status every 30 seconds...")
    while True:
        # Refresh access token inside the loop
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Job State (k={k}): {state}")
        
        if state == "JOB_STATE_SUCCEEDED":
            print(f"Job for k={k} Completed Successfully!")
            return status_data
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
            err_msg = status_data.get("error", "No error details.")
            raise RuntimeError(f"Job ended with state {state}. Details: {err_msg}")
            
        time.sleep(30)

# 6. Download and Parse Results per k-run
def download_and_parse_k_results(configs, job_status_data, test_df, output_parquet_path, k):
    project_id = configs.agent_settings.project_id
    bucket_name = configs.agent_settings.bucket_name
    
    gcs_output_dir = job_status_data.get("outputInfo", {}).get("gcsOutputDirectory")
    if not gcs_output_dir:
        raise ValueError("No gcsOutputDirectory found in job status data.")
        
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
    
    prefix = gcs_output_dir.replace(f"gs://{bucket.name}/", "").replace(f"gs://{bucket.name}", "")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
        
    blobs = list(bucket.list_blobs(prefix=prefix))
    jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]
    
    if not jsonl_blobs:
        raise FileNotFoundError(f"No prediction JSONL files found in GCS prefix: {prefix}")
        
    parsed_data = []
    for blob in jsonl_blobs:
        local_temp = f"temp_k{k}_{os.path.basename(blob.name)}"
        blob.download_to_filename(local_temp)
        
        with open(local_temp, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)
                
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
                
                req_id = None
                try:
                    parts = p['request']['contents'][0]['parts']
                    for part in parts:
                        if 'text' in part and "Sample ID (sclk):" in part['text']:
                            text = part['text']
                            req_id = text.split("|")[0].replace("Sample ID (sclk):", "").strip()
                except Exception:
                    pass
                
                sclk_val = pred_id if pred_id else req_id
                if sclk_val is None:
                    continue
                    
                try:
                    sclk_val = str(int(float(sclk_val)))
                except Exception:
                    sclk_val = str(sclk_val)
                    
                parsed_data.append({
                    "sclk": sclk_val,
                    "predicted_label": pred_class
                })
        os.remove(local_temp)
        
    predictions_df = pd.DataFrame(parsed_data)
    test_lookup = test_df[['sclk', 'class']].copy()
    test_lookup['sclk'] = test_lookup['sclk'].astype(str)
    
    merged_df = pd.merge(predictions_df, test_lookup, on='sclk', how='inner')
    merged_df.rename(columns={"class": "true_label"}, inplace=True)
    merged_df = merged_df[['sclk', 'true_label', 'predicted_label']]
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    merged_df.to_parquet(output_parquet_path)
    print(f"Saved results to {output_parquet_path}")
    return merged_df

# 7. Metrics and Confusion Matrix plotting per k-run
def plot_and_save_k_results(results_df, output_dir, k):
    os.makedirs(output_dir, exist_ok=True)
    
    sub_df = results_df[results_df['predicted_label'] != 'error']
    if len(sub_df) == 0:
        print(f"k={k}: No valid predictions to plot.")
        return 0.0
        
    acc = accuracy_score(sub_df['true_label'], sub_df['predicted_label'])
    print(f"k={k}-shot learning: Accuracy = {acc:.2%}")
    
    labels = sorted(results_df['true_label'].unique())
    cm = confusion_matrix(sub_df['true_label'], sub_df['predicted_label'], labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix (k={k} shots) - Acc: {acc:.2%}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    cm_path = os.path.join(output_dir, f"confusion_matrix_k{k}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    return acc

async def run_incremental_ablation_study():
    configs = Config()
    client = genai.Client()
    model_id = configs.agent_settings.model
    
    full_df = preprocess_huggingface_data("H")
    print(f"Total available rows: {len(full_df)}")
    
    # Select or load reproducible training SCLKs
    train_sclks = get_or_create_train_sclks(full_df)
    
    # Generate VLM explanations pool
    explanations_pool = await generate_explanations_pool(client, model_id, full_df, train_sclks)
    
    # Exclude training SCLKs from testing pool
    all_train_sclks_list = []
    for cls, sclks in train_sclks.items():
        all_train_sclks_list.extend(sclks)
    test_pool = full_df[~full_df['sclk'].isin(all_train_sclks_list)].copy()
    
    # Sample up to 100 unknown test samples per class
    TEST_SIZE_PER_CLASS = 100
    test_dfs = []
    for cls, group in test_pool.groupby('class'):
        n_samples = min(TEST_SIZE_PER_CLASS, len(group))
        if n_samples > 0:
            test_dfs.append(group.sample(n=n_samples, random_state=42))
            print(f"Included class '{cls}' in test set with {n_samples} samples.")
        else:
            print(f"Skipping class '{cls}': no remaining samples.")
            
    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"Test Set Size: {len(test_df)} (up to {TEST_SIZE_PER_CLASS} per class)")
    
    # Incremental k loop
    k_values = [1, 2, 4, 8, 12, 16, 24, 32, 48, 96, 128]
    results_dir = "ablation_results_incremental"
    os.makedirs(results_dir, exist_ok=True)
    
    accuracies = []
    for k in k_values:
        print(f"\n{'='*50}\nNOW WORKING ON k={k} SHOT BATCH PROCESS\n{'='*50}")
        
        # 1. Create requests JSONL
        local_jsonl = f"{results_dir}/requests_k{k}.jsonl"
        create_k_shot_batch_input_file(test_df, explanations_pool, k, local_jsonl)
        
        # 2. Upload, submit job and poll
        job_status = submit_and_poll_batch_job(configs, local_jsonl, k)
        
        # 3. Download and parse results
        output_parquet = f"{results_dir}/results_k{k}.parquet"
        k_results_df = download_and_parse_k_results(configs, job_status, test_df, output_parquet, k)
        
        # 4. Generate metrics and plot CM
        acc = plot_and_save_k_results(k_results_df, results_dir, k)
        accuracies.append(acc)
        
    # Plot final accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, color='darkblue')
    for x, y in zip(k_values, accuracies):
        plt.text(x, y + 0.01, f"{y:.1%}", ha='center', va='bottom', fontsize=9, fontweight='semibold')
    plt.title("Incremental Ablation Study: Accuracy vs. Number of Shots (k)")
    plt.xlabel("Shots (k)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)
    
    curve_path = os.path.join(results_dir, "accuracy_vs_k.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"\nFinal accuracy curve saved to {curve_path}")

if __name__ == "__main__":
    asyncio.run(run_incremental_ablation_study())
