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

async def classify_spectrum(client, model_id, row, k_shot_examples, semaphore):
    async with semaphore:
        img_bytes = generate_spectrum_image_bytes(np.array(row['spectrum']), title=f"Sample {row['sclk']}")
        
        parts = []
        parts.append(types.Part.from_text(text=CLASSIFICATION_USER_PROMPT))
        
        if k_shot_examples:
            parts.append(types.Part.from_text(text="Here are reference examples:"))
            for ex in k_shot_examples:
                parts.append(types.Part.from_text(text=f"Example: {ex['label']} ({ex['explanation']})"))
                parts.append(types.Part.from_bytes(data=ex['image'], mime_type="image/png"))
            parts.append(types.Part.from_text(text="Now, analyze the following spectrum:"))
        
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        parts.append(types.Part.from_text(text=f"Sample ID (sclk): {row['sclk']}"))
        
        contents = [types.Content(role="user", parts=parts)]
        
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION_TEXT,
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "id": types.Schema(type=types.Type.STRING, description="The ID of the run (sclk)"),
                    "class": types.Schema(type=types.Type.STRING, description="The predicted class label"),
                    "explanation": types.Schema(type=types.Type.STRING, description="Explanation for the prediction")
                },
                required=["id", "class", "explanation"]
            )
        )
        
        retries = 3
        for attempt in range(retries):
            try:
                response = await client.aio.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=config
                )
                res_dict = json.loads(response.text)
                return {
                    "sclk": row['sclk'],
                    "true_label": row['class'],
                    "predicted_label": res_dict.get("class", "error"),
                    "explanation": res_dict.get("explanation", "")
                }
            except Exception as e:
                print(f"Classification failed for (sclk: {row['sclk']}): {e}. Retrying {attempt+1}/{retries}...")
                await asyncio.sleep(2 ** attempt)
                
        return {
            "sclk": row['sclk'],
            "true_label": row['class'],
            "predicted_label": "error",
            "explanation": "Failed to classify after retries."
        }

def preprocess_huggingface_data(dataset_type="both"):
    print(f"Loading processed {dataset_type} dataset(s) generated by DataFetchAndParseAgent...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_L_path = os.path.join(base_dir, "cda_dust_agent/data/raw/cda_train_L.parquet")
    train_H_path = os.path.join(base_dir, "cda_dust_agent/data/raw/cda_train_H.parquet")
    
    dfs = []
    if dataset_type in ["L", "both"] and os.path.exists(train_L_path):
        dfs.append(pd.read_parquet(train_L_path))
    if dataset_type in ["H", "both"] and os.path.exists(train_H_path):
        dfs.append(pd.read_parquet(train_H_path))
        
    if not dfs:
        raise FileNotFoundError(f"Processed data files for '{dataset_type}' not found in cda_dust_agent/data/raw/. Run the DataFetchAndParseAgent first.")
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df[full_df['class'] != '?'].copy()
    
    return full_df

async def generate_nested_explanations_pool(client, model_id, pool_df, max_k=32, max_train_map=None):
    """
    Samples up to max_k examples per class once, and generates explanations online concurrently.
    This provides a static, nested candidate pool of few-shot examples.
    """
    if max_train_map is None:
        max_train_map = {
            '5': 16,
            '5-Na': 16
        }
        
    # Sample training samples first and build train_sclks and train_df
    train_sclks = {}
    sampled_rows = []
    for cls in pool_df['class'].unique():
        cls_pool = pool_df[pool_df['class'] == cls]
        k_actual = min(max_k, max_train_map.get(cls, max_k), len(cls_pool))
        
        class_samples = cls_pool.sample(n=k_actual, random_state=123)
        train_sclks[cls] = class_samples['sclk'].tolist()
        sampled_rows.append(class_samples)
        
    train_df = pd.concat(sampled_rows, ignore_index=True)
    
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
                other_sclk = random.choice(other_sclks)
                references.append((other_cls, image_cache[other_sclk]))
                
        task = generate_explanation(client, model_id, row['spectrum'], row['class'], row['sclk'], api_semaphore, references=references)
        explanation_tasks.append(task)
            
    print(f"Generating {len(explanation_tasks)} reference explanations online with dynamic contrastive references...")
    results = await asyncio.gather(*explanation_tasks)
    
    explanations_pool = [r for r in results if r is not None]
    print(f"Successfully generated {len(explanations_pool)} explanations.")
    return explanations_pool

def create_ablation_batch_input_file(test_df, explanations_pool, k_values, output_file):
    """
    Generates a consolidated batch request JSONL containing requests for all k values.
    Uses pre-rendering and image caching with a single reusable Matplotlib figure to prevent OOM.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Generating consolidated batch request file for k={k_values}...")
    
    # 1. Pre-render and cache all test spectrum images (once per sclk)
    print("Pre-rendering and caching target spectrum images...")
    test_image_cache = {}
    
    # Re-use a single figure and axes to prevent memory leaks
    fig, ax = plt.subplots(figsize=(12, 6), dpi=60)
    
    for idx, row in test_df.reset_index(drop=True).iterrows():
        sclk = row['sclk']
        if sclk not in test_image_cache:
            ax.clear()
            ax.plot(row['spectrum'], color='black', linewidth=1.5)
            ax.set_title(f"Sample {sclk}")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0, 630)
            ax.grid(True)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_bytes = buf.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            test_image_cache[sclk] = img_b64
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_df):
            print(f"  -> Pre-rendered {idx + 1}/{len(test_df)} images...")
            
    plt.close(fig)
    
    requests = []
    
    class_explanations = {}
    for ex in explanations_pool:
        cls = ex['label']
        if cls not in class_explanations:
            class_explanations[cls] = []
        class_explanations[cls].append(ex)
        
    for k in k_values:
        print(f"  -> Building request payloads for k={k} shots...")
        k_shot_examples = []
        for cls, exs in class_explanations.items():
            k_shot_examples.extend(exs[:k])
            
        used_sclk_ids = [ex['sclk'] for ex in k_shot_examples]
        
        few_shot_parts = [{"text": CLASSIFICATION_USER_PROMPT}]
        if k_shot_examples:
            few_shot_parts.append({"text": "Here are reference examples:"})
            for ex in k_shot_examples:
                few_shot_parts.append({"text": f"Example: {ex['label']} ({ex['explanation']})"})
                few_shot_parts.append({"inline_data": {"mime_type": "image/png", "data": ex['image_base64']}})
            few_shot_parts.append({"text": "Now, analyze the following spectrum:"})
            
        for _, row in test_df.iterrows():
            if row['sclk'] in used_sclk_ids:
                continue
                
            img_b64 = test_image_cache[row['sclk']]
            
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
            
    print(f"Generated consolidated JSONL file with {len(requests)} requests at {output_file}")
    return output_file

def submit_and_poll_batch_job(configs, local_jsonl_path):
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
                desc="Uploading to GCS",
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
    job_display_name = f"cda-ablation-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
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
    
    req_file = f"batch_request_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(req_file, "w") as f:
        json.dump(global_batch_req, f)
        
    curl_command = [
        "curl", "-s", "-X", "POST",
        f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/batchPredictionJobs",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json; charset=utf-8",
        "-d", f"@{req_file}"
    ]
    
    print("Submitting Batch Job to Vertex AI...")
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
        # Refresh access token inside the loop as batch jobs can run for hours
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
            print("Job Completed Successfully!")
            return status_data
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
            err_msg = status_data.get("error", "No error details.")
            raise RuntimeError(f"Job ended with state {state}. Details: {err_msg}")
            
        time.sleep(30)

def download_and_parse_batch_results(configs, job_status_data, test_df, output_parquet_path):
    project_id = configs.agent_settings.project_id
    bucket_name = configs.agent_settings.bucket_name
    
    gcs_output_dir = job_status_data.get("outputInfo", {}).get("gcsOutputDirectory")
    if not gcs_output_dir:
        raise ValueError("No gcsOutputDirectory found in job status data.")
        
    print(f"Output directory in GCS: {gcs_output_dir}")
    
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
    
    prefix = gcs_output_dir.replace(f"gs://{bucket.name}/", "").replace(f"gs://{bucket.name}", "")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
        
    blobs = list(bucket.list_blobs(prefix=prefix))
    jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]
    
    if not jsonl_blobs:
        raise FileNotFoundError(f"No prediction JSONL files found in GCS prefix: {prefix}")
        
    print(f"Found {len(jsonl_blobs)} prediction JSONL files. Downloading and parsing...")
    
    parsed_data = []
    
    for blob in jsonl_blobs:
        local_temp = f"temp_{os.path.basename(blob.name)}"
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
                pred_k = parsed.get("k")
                pred_class = parsed.get("class") or parsed.get("class_label") or "Noise"
                
                req_id = None
                req_k = None
                try:
                    parts = p['request']['contents'][0]['parts']
                    for part in parts:
                        if 'text' in part and "Sample ID (sclk):" in part['text']:
                            text = part['text']
                            parts_split = text.split("|")
                            req_id = parts_split[0].replace("Sample ID (sclk):", "").strip()
                            if len(parts_split) > 1 and "k:" in parts_split[1]:
                                req_k = int(parts_split[1].replace("k:", "").strip())
                except Exception:
                    pass
                
                sclk_val = pred_id if pred_id else req_id
                k_val = pred_k if pred_k is not None else req_k
                
                if sclk_val is None:
                    print(f"Warning: Could not resolve sclk for instance.")
                    continue
                    
                try:
                    sclk_val = str(int(float(sclk_val)))
                except Exception:
                    sclk_val = str(sclk_val)
                    
                try:
                    k_val = int(k_val)
                except Exception:
                    k_val = 0
                
                parsed_data.append({
                    "sclk": sclk_val,
                    "predicted_label": pred_class,
                    "k": k_val
                })
                
        os.remove(local_temp)
        
    predictions_df = pd.DataFrame(parsed_data)
    
    test_lookup = test_df[['sclk', 'class']].copy()
    test_lookup['sclk'] = test_lookup['sclk'].astype(str)
    
    merged_df = pd.merge(predictions_df, test_lookup, on='sclk', how='inner')
    merged_df.rename(columns={"class": "true_label"}, inplace=True)
    merged_df = merged_df[['sclk', 'true_label', 'predicted_label', 'k']]
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    merged_df.to_parquet(output_parquet_path)
    print(f"Saved all results ({len(merged_df)} rows) to consolidated Parquet file at {output_parquet_path}")
    return merged_df

def plot_and_save_ablation_results(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    k_values = sorted(results_df['k'].unique())
    accuracies = []
    
    print("\n--- Ablation Study Results ---")
    for k in k_values:
        sub_df = results_df[(results_df['k'] == k) & (results_df['predicted_label'] != 'error')]
        if len(sub_df) == 0:
            print(f"k={k}: No valid predictions.")
            accuracies.append(0.0)
            continue
            
        acc = accuracy_score(sub_df['true_label'], sub_df['predicted_label'])
        accuracies.append(acc)
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
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, color='darkblue')
    for x, y in zip(k_values, accuracies):
        plt.text(x, y + 0.01, f"{y:.1%}", ha='center', va='bottom', fontsize=9, fontweight='semibold')
    plt.title("Ablation Study: Accuracy vs. Number of Shots (k)")
    plt.xlabel("Shots (k)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)
    
    curve_path = os.path.join(output_dir, "accuracy_vs_k.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved accuracy curve plot to {curve_path}")

async def run_ablation_study():
    k_values = [1, 2, 4, 8, 16, 32, 64]
    
    config = Config()
    client = genai.Client()
    model_id = config.agent_settings.model
    
    full_df = preprocess_huggingface_data("H")
    print(f"Total labeled, filtered rows available for ablation: {len(full_df)}")
    
    api_semaphore = asyncio.Semaphore(5)
    
    max_train_map = {
        '5': 16,
        '5-Na': 16
    }
    
    for k in k_values:
        print(f"\n{'='*40}\nStarting ablation study for k={k}\n{'='*40}")
        output_dir = f"cda_dust_agent/data/ablation_study/k_{k}"
        os.makedirs(output_dir, exist_ok=True)
        results_csv_path = os.path.join(output_dir, f"results_k{k}.csv")
        
        class_counts = full_df['class'].value_counts()
        
        min_test_samples = 1 
        viable_classes = []
        for cls, count in class_counts.items():
            k_actual = min(k, max_train_map.get(cls, k))
            if count >= (k_actual + min_test_samples):
                viable_classes.append(cls)
                
        if not viable_classes:
            print(f"No viable classes found for k={k}. Skipping.")
            continue
            
        print(f"Viable classes for k={k}: {viable_classes}")
        
        k_shot_examples = []
        used_sclks = []
        
        explanation_tasks = []
        
        for cls in viable_classes:
            k_actual = min(k, max_train_map.get(cls, k))
            class_pool = full_df[full_df['class'] == cls].sample(n=k_actual, random_state=42)
            
            for _, row in class_pool.iterrows():
                task = generate_explanation(client, model_id, row['spectrum'], row['class'], row['sclk'], api_semaphore)
                explanation_tasks.append(task)
                used_sclks.append(row['sclk'])
        
        print(f"Generating {len(explanation_tasks)} explanations concurrently for k={k}...")
        results = await asyncio.gather(*explanation_tasks)
        for res in results:
            if res:
                k_shot_examples.append(res)
                
        print(f"Successfully generated {len(k_shot_examples)} explanations for k={k}.")
        
        batch_test_df = full_df[full_df['class'].isin(viable_classes)].copy()
        batch_test_df = batch_test_df[~batch_test_df['sclk'].isin(used_sclks)]
        
        print(f"Executing inference on {len(batch_test_df)} test spectra for k={k}...")
        
        inference_tasks = []
        for _, row in batch_test_df.iterrows():
            task = classify_spectrum(client, model_id, row, k_shot_examples, api_semaphore)
            inference_tasks.append(task)
            
        all_results = []
        chunk_size = 500
        for i in range(0, len(inference_tasks), chunk_size):
            chunk = inference_tasks[i:i + chunk_size]
            print(f"Processing inference chunk {i} to {i+len(chunk)-1} / {len(inference_tasks)}...")
            chunk_results = await asyncio.gather(*chunk)
            all_results.extend(chunk_results)
            
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(results_csv_path, index=False)
                
        print(f"Saved inference results for k={k} to {results_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CDA Ablation Study.")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode using Vertex AI Batch Prediction API.")
    args = parser.parse_args()
    
    if args.batch:
        async def main_batch():
            configs = Config()
            client = genai.Client()
            model_id = configs.agent_settings.model
            
            full_df = preprocess_huggingface_data("H")
            print(f"Total available rows: {len(full_df)}")
            
            TEST_SIZE_PER_CLASS = 100
            test_dfs = []
            pool_dfs = []
            
            for cls, group in full_df.groupby('class'):
                if len(group) >= TEST_SIZE_PER_CLASS + 1:
                    group = group.sample(frac=1, random_state=42).reset_index(drop=True)
                    test_dfs.append(group.head(TEST_SIZE_PER_CLASS))
                    pool_dfs.append(group.tail(len(group) - TEST_SIZE_PER_CLASS))
                else:
                    print(f"Skipping class '{cls}' - not enough samples ({len(group)})")
                    
            test_df = pd.concat(test_dfs, ignore_index=True)
            pool_df = pd.concat(pool_dfs, ignore_index=True)
            
            print(f"Test Set Size: {len(test_df)} (up to {TEST_SIZE_PER_CLASS} per class)")
            print(f"Remaining Pool Size: {len(pool_df)}")
            
            print("\n--- STEP 1: Pre-generating Nested Explanations Pool ---")
            explanations_pool = await generate_nested_explanations_pool(client, model_id, pool_df)
            
            print("\n--- STEP 2: Creating Consolidated Batch Requests JSONL File ---")
            k_values = [1, 2, 4, 8, 12, 16, 24, 32]
            local_jsonl = "cda_dust_agent/data/input/ablation_requests.jsonl"
            create_ablation_batch_input_file(test_df, explanations_pool, k_values, local_jsonl)
            
            print("\n--- STEP 3: Submitting Batch Job to Vertex AI & Polling ---")
            job_status = submit_and_poll_batch_job(configs, local_jsonl)
            
            print("\n--- STEP 4: Downloading and Parsing Batch Prediction Results ---")
            output_parquet = "cda_dust_agent/data/results/all_ablation_results.parquet"
            results_df = download_and_parse_batch_results(configs, job_status, test_df, output_parquet)
            
            print("\n--- STEP 5: Generating Metrics & Confusion Matrix Files ---")
            plot_and_save_ablation_results(results_df, "ablation_results")
            
        asyncio.run(main_batch())
    else:
        asyncio.run(run_ablation_study())