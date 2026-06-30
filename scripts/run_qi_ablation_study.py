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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cda_dust_agent.config import Config
from cda_dust_agent.prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT
from cda_dust_agent.tools.utils import generate_spectrum_image_bytes, parse_response
from scripts.run_ablation_study_incremental import (
    preprocess_huggingface_data,
    generate_explanation
)

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
        blob.chunk_size = 10 * 1024 * 1024
        blob.upload_from_file(wrapped_file, content_type="application/json", timeout=600)
        
    gcs_source = f"gs://{bucket.name}/{gcs_input_path}"
    print(f"Uploaded to {gcs_source}")
    
    access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
    job_display_name = f"cda-qi-ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
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
    
    req_file = f"batch_request_qi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        local_temp = f"temp_qi_{os.path.basename(blob.name)}"
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
                
                pred_class = parsed.get("class") or parsed.get("class_label") or "Noise"
                pred_id = parsed.get("id")
                bin_idx = parsed.get("bin")
                run_idx = parsed.get("run")
                
                # Fallback to request text if response JSON didn't include them
                if pred_id is None or bin_idx is None or run_idx is None:
                    try:
                        parts = p['request']['contents'][0]['parts']
                        for part in parts:
                            if 'text' in part and "Sample ID (sclk):" in part['text']:
                                text = part['text']
                                meta_parts = text.split("|")
                                if pred_id is None:
                                    pred_id = meta_parts[0].replace("Sample ID (sclk):", "").strip()
                                if bin_idx is None:
                                    bin_idx = int(meta_parts[1].replace("bin:", "").strip())
                                if run_idx is None:
                                    run_idx = int(meta_parts[2].replace("run:", "").strip())
                    except Exception:
                        pass
                
                if pred_id is None:
                    continue
                    
                try:
                    pred_id = str(int(float(pred_id)))
                except Exception:
                    pred_id = str(pred_id)
                
                parsed_data.append({
                    "sclk": pred_id,
                    "predicted_label": pred_class,
                    "bin": bin_idx,
                    "run": run_idx
                })
        os.remove(local_temp)
        
    predictions_df = pd.DataFrame(parsed_data)
    test_lookup = test_df[['sclk', 'class']].copy()
    test_lookup['sclk'] = test_lookup['sclk'].astype(str)
    
    merged_df = pd.merge(predictions_df, test_lookup, on='sclk', how='inner')
    merged_df.rename(columns={"class": "true_label"}, inplace=True)
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    merged_df.to_parquet(output_parquet_path)
    print(f"Saved consolidated results to: {output_parquet_path}")
    return merged_df

def generate_statistics_and_plots(results_df, output_dir, bin_edges):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    cm_dir = os.path.join(plots_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    
    bin_stats = {}
    
    # Analyze each bin
    for b in sorted(results_df['bin'].unique()):
        bin_min, bin_max = bin_edges[b], bin_edges[b+1]
        bin_df = results_df[results_df['bin'] == b]
        
        run_accuracies = []
        for r in sorted(bin_df['run'].unique()):
            run_df = bin_df[bin_df['run'] == r]
            valid_df = run_df[run_df['predicted_label'] != 'error']
            
            if len(valid_df) == 0:
                continue
                
            acc = accuracy_score(valid_df['true_label'], valid_df['predicted_label'])
            run_accuracies.append(acc)
            
            # Save confusion matrix
            labels = sorted(run_df['true_label'].unique())
            cm = confusion_matrix(valid_df['true_label'], valid_df['predicted_label'], labels=labels)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title(f"Bin {b} (Log10(QI): [{bin_min:.2f}, {bin_max:.2f})) - Run {r} - Acc: {acc:.2%}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            
            cm_path = os.path.join(cm_dir, f"bin_{b}_run_{r}.png")
            plt.savefig(cm_path)
            plt.close()
            
        mean_acc = np.mean(run_accuracies) if run_accuracies else 0.0
        std_acc = np.std(run_accuracies) if run_accuracies else 0.0
        
        bin_stats[int(b)] = {
            "log_qi_range": [float(bin_min), float(bin_max)],
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "run_accuracies": [float(x) for x in run_accuracies]
        }
        print(f"Bin {b} ([{bin_min:.2f}, {bin_max:.2f})): Mean Acc = {mean_acc:.2%} (std: {std_acc:.2%})")
        
    # Save statistics JSON
    stats_path = os.path.join(output_dir, "bin_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(bin_stats, f, indent=4)
    print(f"Saved statistics JSON to {stats_path}")
    
    # Plot final curve with error bars
    bins_list = sorted(list(bin_stats.keys()))
    mean_accs = [bin_stats[b]['mean_accuracy'] for b in bins_list]
    std_accs = [bin_stats[b]['std_accuracy'] for b in bins_list]
    bin_labels = [f"B{b}\n[{bin_stats[b]['log_qi_range'][0]:.2f}, {bin_stats[b]['log_qi_range'][1]:.2f})" for b in bins_list]
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(bins_list, mean_accs, yerr=std_accs, fmt='-o', color='darkblue', ecolor='red', elinewidth=2, capsize=4, linewidth=2, label='Mean Accuracy')
    
    # Annotate mean values
    for x, y in zip(bins_list, mean_accs):
        plt.text(x, y + 0.015, f"{y:.1%}", ha='center', va='bottom', fontsize=9, fontweight='semibold')
        
    plt.title("QI-Amplitude Ablation Study: k=24 Accuracy vs. QI-amplitude Bin")
    plt.xlabel("QI Amplitude Bin (log10(QI_ampl))")
    plt.ylabel("Inference Accuracy")
    plt.xticks(bins_list, bin_labels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.legend()
    
    curve_path = os.path.join(plots_dir, "qi_ablation_accuracy_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved final accuracy curve plot to {curve_path}")

async def run_qi_ablation_study(use_cached_explanations=True):
    configs = Config()
    client = genai.Client()
    model_id = configs.agent_settings.model
    
    full_df = preprocess_huggingface_data("H")
    print(f"Total available rows: {len(full_df)}")
    
    # Calculate log_qi for all rows (excluding Noise)
    full_df['log_qi'] = np.log10(full_df['qi_ampl'])
    
    # Determine bin boundaries based on non-noise range in dataset
    non_noise_df = full_df[full_df['class'] != 'Noise'].copy()
    min_log_qi = non_noise_df['log_qi'].min()
    max_log_qi = non_noise_df['log_qi'].max()
    
    print(f"Non-noise log10(QI) range in dataset: {min_log_qi:.4f} to {max_log_qi:.4f}")
    bin_edges = np.linspace(min_log_qi, max_log_qi, 6)
    
    # Step 1: Select few-shot SCLKs for all 5 bins (1 run per bin)
    # We want k=24 for classes 1, 2, 3, 4, Noise and k=16 for 5, 5-Na
    few_shot_by_bin = {}
    all_selected_sclks = set()
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Load training SCLKs from incremental study configurations to ensure consistency for non-filtered classes
    from scripts.run_ablation_study_incremental import get_or_create_train_sclks
    train_sclks = get_or_create_train_sclks(full_df)
    
    max_k_map = {
        '1': 24,
        '2': 24,
        '3': 24,
        '4': 24,
        'Noise': 24,
        '5': 16,
        '5-Na': 16
    }
    
    for b in range(5):
        bin_min, bin_max = bin_edges[b], bin_edges[b+1]
        bin_few_shot = {}
        
        for cls, group in full_df.groupby('class'):
            cls_str = str(cls)
            k_target = max_k_map[cls_str]
            
            if cls_str in ['Noise', '5', '5-Na']:
                # No QI-filtering: use the exact fixed training SCLKs from the incremental study
                bin_few_shot[cls_str] = train_sclks[cls_str]
                all_selected_sclks.update(train_sclks[cls_str])
            else:
                # Filter by bin
                bin_group = group[(group['log_qi'] >= bin_min) & (group['log_qi'] < bin_max)]
                if len(bin_group) >= k_target:
                    sampled_group = bin_group.sample(n=k_target, random_state=42 + b)
                else:
                    # Fallback: take all in bin, then fill with closest by log_qi distance
                    bin_center = (bin_min + bin_max) / 2
                    group = group.copy()
                    group['dist_to_center'] = (group['log_qi'] - bin_center).abs()
                    # Exclude the ones already in bin_group
                    remaining = group[~group['sclk'].isin(bin_group['sclk'])]
                    n_more = k_target - len(bin_group)
                    closest_more = remaining.nsmallest(n_more, 'dist_to_center')
                    sampled_group = pd.concat([bin_group, closest_more])
                    
                bin_few_shot[cls_str] = sampled_group['sclk'].tolist()
                all_selected_sclks.update(sampled_group['sclk'].tolist())
        
        few_shot_by_bin[b] = bin_few_shot
        
    print(f"Total unique few-shot SCLKs selected across all 5 bins: {len(all_selected_sclks)}")
    
    # Step 2: Generate explanations pool for chosen SCLKs
    results_dir = "qi_ablation_results"
    os.makedirs(results_dir, exist_ok=True)
    cache_path = os.path.join(results_dir, "explanations_cache.json")
    
    explanations_pool = []
    cached_explanations = {}
    
    if use_cached_explanations and os.path.exists(cache_path):
        print(f"Loading cached explanations from {cache_path}...")
        try:
            with open(cache_path, 'r') as f:
                cached_explanations = json.load(f)
            print(f"Loaded {len(cached_explanations)} explanations from cache.")
        except Exception as e:
            print(f"Error loading cache: {e}. Will regenerate all explanations.")
            
    # Find SCLKs that are missing from cache
    missing_sclks = [sclk for sclk in all_selected_sclks if str(sclk) not in cached_explanations]
    
    if len(missing_sclks) > 0:
        print(f"Generating reference explanations for {len(missing_sclks)} missing SCLKs...")
        # Pre-render and cache few-shot images for the selected training spectra
        print("Pre-rendering and caching few-shot spectrum images...")
        image_cache = {}
        selected_df = full_df[full_df['sclk'].isin(all_selected_sclks)].copy()
        for _, row in selected_df.iterrows():
            sclk = row['sclk']
            cls = row['class']
            label = f"Class {cls}" if str(cls).lower() != 'noise' else "Noise"
            img_bytes = generate_spectrum_image_bytes(np.array(row['spectrum']), title=f"{label} Reference Sample {sclk}")
            image_cache[sclk] = img_bytes
            
        api_semaphore = asyncio.Semaphore(10)
        explanation_tasks = []
        
        selected_by_class = {cls: group['sclk'].tolist() for cls, group in selected_df.groupby('class')}
        
        # Only request explanation for the missing SCLKs
        missing_df = full_df[full_df['sclk'].isin(missing_sclks)].copy()
        for _, row in missing_df.iterrows():
            target_sclk = row['sclk']
            target_cls = str(row['class'])
            
            # Pick one random reference SCLK from each of the other classes
            references = []
            for other_cls, other_sclks in selected_by_class.items():
                other_cls_str = str(other_cls)
                if other_cls_str != target_cls:
                    other_sclk = random.choice(other_sclks)
                    references.append((other_cls_str, image_cache[other_sclk]))
                    
            task = generate_explanation(client, model_id, row['spectrum'], row['class'], row['sclk'], api_semaphore, references=references)
            explanation_tasks.append(task)
            
        new_results = await asyncio.gather(*explanation_tasks)
        
        for r in new_results:
            if r is not None:
                # Remove the raw bytes key 'image' for JSON serialization compatibility
                r_copy = r.copy()
                if 'image' in r_copy:
                    del r_copy['image']
                cached_explanations[str(r['sclk'])] = r_copy
                
        # Save cache back
        with open(cache_path, 'w') as f:
            json.dump(cached_explanations, f, indent=4)
        print(f"Saved updated explanations cache to {cache_path}.")
    else:
        print("All selected few-shot SCLKs found in cache. Skipping explanations generation.")
        
    # Populate the explanations pool from the cache
    for sclk in all_selected_sclks:
        sclk_str = str(sclk)
        if sclk_str in cached_explanations:
            explanations_pool.append(cached_explanations[sclk_str])
    print(f"Successfully compiled explanations pool of size {len(explanations_pool)}.")
    
    # Step 3: Build consolidated requests file
    results_dir = "qi_ablation_results"
    os.makedirs(results_dir, exist_ok=True)
    local_jsonl = os.path.join(results_dir, "requests_qi_ablation.jsonl")
    
    # Pre-render and cache target test spectrum images
    print("Pre-rendering target test spectrum images...")
    test_image_cache = {}
    fig, ax = plt.subplots(figsize=(12, 6), dpi=60)
    for idx, row in full_df.iterrows():
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
            test_image_cache[sclk] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    exp_by_sclk = {str(ex['sclk']): ex for ex in explanations_pool}
    total_requests_written = 0
    
    with open(local_jsonl, 'w') as out_f:
        for b in range(5):
            bin_few_shot = few_shot_by_bin[b]
            bin_few_shot_sclks_list = []
            for sclks in bin_few_shot.values():
                bin_few_shot_sclks_list.extend(sclks)
                
            # Exclude these few-shot SCLKs from the test set for this bin
            test_pool = full_df[~full_df['sclk'].isin(bin_few_shot_sclks_list)].copy()
            
            # Sample up to 100 test samples per class
            TEST_SIZE_PER_CLASS = 100
            test_dfs = []
            for cls, group in test_pool.groupby('class'):
                n_samples = min(TEST_SIZE_PER_CLASS, len(group))
                if n_samples > 0:
                    test_dfs.append(group.sample(n=n_samples, random_state=42))
            test_df = pd.concat(test_dfs, ignore_index=True)
            
            # Build few-shot examples parts
            k_examples = []
            for cls_str, sclks in bin_few_shot.items():
                for sclk_val in sclks:
                    sclk_val_str = str(sclk_val)
                    if sclk_val_str in exp_by_sclk:
                        k_examples.append(exp_by_sclk[sclk_val_str])
            
            few_shot_parts = [{"text": CLASSIFICATION_USER_PROMPT}]
            if k_examples:
                few_shot_parts.append({"text": "Here are reference examples:"})
                for ex in k_examples:
                    few_shot_parts.append({"text": f"Example: {ex['label']} ({ex['explanation']})"})
                    few_shot_parts.append({"inline_data": {"mime_type": "image/png", "data": ex['image_base64']}})
                few_shot_parts.append({"text": "Now, analyze the following spectrum:"})
                
            for _, row in test_df.iterrows():
                img_b64 = test_image_cache[row['sclk']]
                current_parts = few_shot_parts.copy()
                current_parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
                current_parts.append({"text": f"Sample ID (sclk): {row['sclk']} | bin: {b} | run: 0"})
                
                request = {
                    "request": {
                        "contents": [{"role": "user", "parts": current_parts}],
                        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION_TEXT}]},
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "responseSchema": {
                                "type": "OBJECT",
                                "properties": {
                                    "id": {"type": "STRING", "description": "The ID of the run (sclk)"},
                                    "class": {"type": "STRING", "description": "The predicted class label"},
                                    "explanation": {"type": "STRING", "description": "Explanation for the prediction"},
                                    "bin": {"type": "INTEGER", "description": "The bin number representing the QI-amplitude range used (0 to 4)"},
                                    "run": {"type": "INTEGER", "description": "The run number of the independent trials (always 0)"}
                                },
                                "required": ["id", "class", "explanation", "bin", "run"]
                            }
                        }
                    }
                }
                out_f.write(json.dumps(request) + '\n')
                total_requests_written += 1
                
    print(f"Consolidated request file created successfully with {total_requests_written} requests.")
    
    # Step 4: Submit Batch Job and Poll
    job_status = submit_and_poll_batch_job(configs, local_jsonl)
    
    # Step 5: Download and parse GCS output
    output_parquet = os.path.join(results_dir, "results_qi_ablation.parquet")
    results_df = download_and_parse_batch_results(configs, job_status, full_df[['sclk', 'class']], output_parquet)
    
    # Step 6: Calculate statistics and save plots
    generate_statistics_and_plots(results_df, results_dir, bin_edges)

if __name__ == "__main__":
    asyncio.run(run_qi_ablation_study())
