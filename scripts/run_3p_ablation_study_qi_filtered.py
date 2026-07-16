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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.signal import savgol_filter
import huggingface_hub
import tqdm
import gc

warnings.filterwarnings("ignore")
import logging
logging.getLogger("google_genai").setLevel(logging.WARNING)

from dotenv import load_dotenv
try:
    dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
    project_root = os.path.join(os.path.dirname(__file__), '..')
except NameError:
    dotenv_path = "../.env"
    project_root = ".."

load_dotenv(dotenv_path=dotenv_path)

import sys
sys.path.append(os.path.abspath(project_root))

from cda_dust_agent.config import Config
from cda_dust_agent.tools.utils import generate_spectrum_image_bytes, parse_response

# Load current best prompts from cda_dust_agent.prompts
from cda_dust_agent.prompts import (
    SYSTEM_INSTRUCTION_TEXT,
    SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT,
    GENERAL_PROFILES,
    CLASSIFICATION_USER_PROMPT
)

CLASSIFICATION_USER_PROMPT_SELF_CORRECT = """Here is the first-pass classification for the target spectrum:
- First-pass Predicted Class: {first_pass_class}
- First-pass Explanation: {first_pass_explanation}

Review the target spectrum image carefully against the reference examples of Class 3 and Class 3-P.
Perform a self-correction check. Specifically, verify if the first-pass prediction fell into the trap of misidentifying a Class 3 spectrum with multiple broad peaks or baseline noise as Class 3-P.

Return your analysis strictly in this JSON format:
{{
    "id": "{sclk}",
    "first_pass_analysis_critique": "<your critique of the first-pass explanation, explaining if it fell into the Class 3 baseline noise trap>",
    "class": "<your final prediction: '3' or '3-P'>",
    "explanation": "<your final 1-2 sentence explanation supporting the final prediction>"
}}
"""

ANNOTATION_USER_PROMPT = """This is a time-of-flight mass spectrum for a particle belonging to the known class '{label}'.
Please provide a brief, 1-2 sentence description of the key visual features that characterize this spectrum as '{label}'. Do not output JSON, just the text description."""

CONTRASTIVE_ANNOTATION_USER_PROMPT = """You are an expert Cosmic Dust Spectroscopist.
We want you to write a brief, 1-2 sentence description of the key visual features of the Target Spectrum (labeled as Class '{label}') to be used as a reference example for class '{label}'.

To help you write a contrastive explanation that clearly distinguishes '{label}' from all other classes, we have provided the Target Spectrum (Image A) alongside reference spectra from the other classes.

Please review all the provided images:
- Image A (Target): This is the spectrum of '{label}' you must describe.
{reference_descriptions}

Key Spectral Guidelines to remember:
- Class 1: Pure water ice. Clear, sharp sequence of hydronium cluster peaks ($H_3O^+(H_2O)_n$) at mass 19, 37, 55, 73... (index locations ~85, ~120, ~146, ~169, ~189). Global maximum is early (~80-100). Valleys between peaks return fully to baseline.
- Class 2: Organic-rich water ice. Same hydronium peaks as Class 1, but with significant valley-filling, organic background, or intermediate peaks between them. Global maximum is often in the 180-300 range.
- Class 4: Mineral spectrum. Needle-sharp atomic spikes (Mg+, Si+, Fe+) with a quiet baseline, and completely lacks the repeating water-ice cluster sequence. Mid-mass envelopes with global maximum at index 280-350 and a distinct late cluster around index 420-480.
- Class 3: Organics. Multiple broad asymmetric shark-fin peak clusters representing carbon clusters. Crucially, on the scaled [0, 1] y-axis, the valleys between clusters drop significantly lower (down to y < 0.15), and it lacks the massive early maximum (index 200) and the flat, continuous chemical noise plateau past index 400 seen in Class 3-P.
- Class 5: Bimodal extreme (overwhelmingly intense early peak with long trailing decay, rest is dense low-amplitude barcode noise).
- Class 5-Na: Singular sharp Sodium payload peak at index 135-160, followed by a delayed, noisy detector-saturation plateau.
- Class 3-P: Burst-and-trail signature. Massive primary peak complex (index ~200-220) as the global maximum (y = 1.0), a distinct secondary peak at index ~330-345, and an elevated baseline plateau past index 400 that remains highly elevated (y ≈ 0.3 to 0.5) and never returns to zero.
- Noise: Narrow initial trigger spike followed by broad envelope of digitizer noise ("grass"), devoid of chemical peaks.

Write a 1-2 sentence description of the visual features of Image A (Target) that characterize it as '{label}', highlighting specific details (such as peak spacing trend, peak shape, or noise baseline) that help distinguish it from the other classes. Do not output JSON, just return the text description."""

# Memory-safe parser to read predictions directly from GCS line string
def extract_prediction_from_line(line):
    idx = line.rfind('"text":"{')
    if idx == -1:
        idx = line.rfind('"text":"{\\n')
    if idx == -1:
        return None
        
    start_val_idx = idx + 7  # index of the double quote of the string value
    
    end_idx = -1
    pos = start_val_idx + 1
    while pos < len(line):
        if line[pos] == '"':
            # Count backslashes before it
            bs_count = 0
            k = pos - 1
            while k >= start_val_idx and line[k] == '\\':
                bs_count += 1
                k -= 1
            if bs_count % 2 == 0:
                end_idx = pos
                break
        pos += 1
        
    if end_idx == -1:
        return None
        
    escaped_json_str = line[start_val_idx:end_idx+1]
    try:
        json_str = json.loads(escaped_json_str)
        return json.loads(json_str)
    except Exception:
        return None

# 1. Download and Preprocess
def download_and_preprocess_data():
    print("Downloading train dataset from Hugging Face...")
    REPO_ID = "CosmicDustGroup/cassini-cda-spectra"
    FILENAME_TRAIN = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"
    
    file_train_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_TRAIN, repo_type="dataset")
    print(f"Downloaded train dataset to: {file_train_path}")
    
    df = pd.read_parquet(file_train_path)
    print(f"Loaded {len(df)} raw spectra.")
    
    # Filter 1018 length spectra
    df = df[df['spectrum'].apply(len) == 1018].copy()
    print(f"Filtered to {len(df)} spectra of length 1018.")
    
    # Crop spectrum to index 10 to 640
    def crop_spectrum(spectrum):
        return spectrum[10:641]
    df['spectrum'] = df['spectrum'].apply(crop_spectrum)
    
    # Map labels
    def map_label(x):
        if not isinstance(x, str):
            return "?"
        if x in ["?", "X", "2-X"]:
            return "?"
        if x == "3-P":
            return "3-P"
        if x.startswith("3-") or x == "3":
            return "3"
        return x
        
    df['class'] = df['class'].apply(map_label)
    df = df[df['class'] != '?'].copy()
    print(f"Remaining active spectra after mapping: {len(df)}")
    
    # Apply QI Amplitude Filtering before splitting
    # 10 fC to 2 pC
    lower_limit = 10 * 1e-15  # 1e-14 C
    upper_limit = 2 * 1e-12   # 2e-12 C
    
    df = df[(df['qi_ampl'] >= lower_limit) & (df['qi_ampl'] <= upper_limit)].copy()
    print(f"Applied QI filtering (10 fC <= QI <= 2 pC). Remaining spectra: {len(df)}")
    print("Class distribution after filtering:")
    print(df['class'].value_counts())
    
    # Scaling and smoothing
    def qm_scaling_savgol(spectrum):
        spectrum = np.array(spectrum, dtype=float)
        # Apply log10
        log_spec = np.log10(spectrum + np.abs(np.min(spectrum)))
        
        # Replace negative infinity/NaN with minimum finite value
        finite_mask = np.isfinite(log_spec)
        if np.any(finite_mask):
            min_finite_val = np.min(log_spec[finite_mask])
        else:
            min_finite_val = 0
        log_spec = np.nan_to_num(log_spec, neginf=min_finite_val, nan=min_finite_val)
        
        # Min-max scaling
        spec_min = np.min(log_spec)
        spec_max = np.max(log_spec)
        range_val = spec_max - spec_min
        if range_val > 0:
            scaled_spectrum = (log_spec - spec_min) / range_val
        else:
            scaled_spectrum = np.zeros_like(log_spec)
            
        return savgol_filter(scaled_spectrum, 5, 3)
        
    df['spectrum'] = df['spectrum'].apply(qm_scaling_savgol)
    print("Scaling and smoothing complete.")
    
    return df

# 2. Split train/test
def split_train_test(df, max_train_map=None, test_size_per_class=None):
    if max_train_map is None:
        max_train_map = {
            '3-P': 4,
            '5': 16,
            '5-Na': 16
        }
        
    train_sclks = {}
    sampled_rows = []
    
    classes = ['Noise', '1', '2', '3', '4', '5', '5-Na', '3-P']
    
    for cls in classes:
        cls_df = df[df['class'] == cls]
        # Train limit based on map, default is 16
        n_train = max_train_map.get(cls, 16)
        n_train = min(n_train, len(cls_df))
        
        cls_train = cls_df.sample(n=n_train, random_state=123)
        train_sclks[cls] = cls_train['sclk'].tolist()
        sampled_rows.append(cls_train)
        
    train_df = pd.concat(sampled_rows, ignore_index=True)
    all_train_sclks = [sclk for sclks in train_sclks.values() for sclk in sclks]
    
    test_pool = df[~df['sclk'].isin(all_train_sclks)].copy()
    
    test_dfs = []
    for cls in classes:
        cls_test_pool = test_pool[test_pool['class'] == cls]
        if test_size_per_class is not None:
            n_samples = min(test_size_per_class, len(cls_test_pool))
        else:
            n_samples = len(cls_test_pool)
            
        if n_samples > 0:
            cls_test = cls_test_pool.sample(n=n_samples, random_state=42)
            test_dfs.append(cls_test)
            print(f"Included class '{cls}' in test set with {len(cls_test)} samples.")
        
    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"Train samples: {len(train_df)} (classes breakdown: { {k: len(v) for k, v in train_sclks.items()} })")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, test_df, train_sclks

# 3. Generate VLM explanations concurrently
async def generate_explanation(client, model_id, spectrum_array, label, sclk, semaphore, references=None):
    async with semaphore:
        def format_class_label(cls):
            cls_str = str(cls)
            if cls_str.lower().startswith("class"):
                return cls_str
            return f"Class {cls_str}" if cls_str.lower() != 'noise' else "Noise"

        target_label = format_class_label(label)
        img_bytes = generate_spectrum_image_bytes(np.array(spectrum_array), title=f"{target_label} Sample {sclk}")
        
        if references:
            ref_descriptions = ""
            for idx, (other_cls, _) in enumerate(references):
                char_code = chr(66 + idx) # B, C, D, E...
                ref_descriptions += f"- Image {char_code}: Reference Spectrum of '{format_class_label(other_cls)}'\\n"
                
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

async def generate_explanations_pool(client, model_id, train_df, train_sclks):
    print("Pre-rendering and caching training spectrum images...")
    image_cache = {}
    for _, row in train_df.iterrows():
        sclk = row['sclk']
        cls = row['class']
        label = f"Class {cls}" if str(cls).lower() != 'noise' else "Noise"
        img_bytes = generate_spectrum_image_bytes(np.array(row['spectrum']), title=f"{label} Reference Sample {sclk}")
        image_cache[sclk] = img_bytes
        
    api_semaphore = asyncio.Semaphore(10)
    explanation_tasks = []
    
    import random
    random.seed(123)
    
    print(f"Generating explanations for {len(train_df)} training samples concurrently with dynamic contrastive references...")
    for _, row in train_df.iterrows():
        target_sclk = row['sclk']
        target_cls = row['class']
        
        references = []
        for other_cls, other_sclks in train_sclks.items():
            if other_cls != target_cls:
                other_sclk = random.choice(other_sclks)
                references.append((other_cls, image_cache[other_sclk]))
                
        task = generate_explanation(client, model_id, row['spectrum'], row['class'], row['sclk'], api_semaphore, references=references)
        explanation_tasks.append(task)
            
    results = await asyncio.gather(*explanation_tasks)
    explanations_pool = [r for r in results if r is not None]
    print(f"Successfully generated {len(explanations_pool)} explanations.")
    return explanations_pool

# 4. Create single batch file
def create_ablation_batch_input_file(test_df, explanations_pool, k_value, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ablation_requests_3p.jsonl")
    print(f"Generating single batch request file: {output_file}...")
    
    class_explanations = {}
    for ex in explanations_pool:
        cls = ex['label']
        if cls not in class_explanations:
            class_explanations[cls] = []
        class_explanations[cls].append(ex)
        
    print(f"Pre-rendering and caching target spectrum images for {len(test_df)} samples...")
    test_image_cache = {}
    fig, ax = plt.subplots(figsize=(12, 6), dpi=60)
    
    for idx, row in test_df.iterrows():
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
    gc.collect()
    
    k_shot_examples = []
    for cls, exs in class_explanations.items():
        limit = 4 if cls == "3-P" else 16
        k_shot_examples.extend(exs[:limit])
        
    used_sclk_ids = [ex['sclk'] for ex in k_shot_examples]
    
    few_shot_parts = [{"text": CLASSIFICATION_USER_PROMPT}]
    if k_shot_examples:
        few_shot_parts.append({"text": "Here are reference examples:"})
        for ex in k_shot_examples:
            label_key = ex['label']
            gen_profile = GENERAL_PROFILES.get(label_key, '')
            class_label = f"Class {label_key}" if str(label_key).lower() != 'noise' else "Noise"
            few_shot_parts.append({"text": f"Example: {class_label}\\\\n- General Profile: {gen_profile}\\\\n- Specific Sample Features: {ex['explanation']}"})
            few_shot_parts.append({"inline_data": {"mime_type": "image/png", "data": ex['image_base64']}})
        few_shot_parts.append({"text": "Now, analyze the following spectrum:"})
        
    requests = []
    for _, row in test_df.iterrows():
        if row['sclk'] in used_sclk_ids:
            continue
            
        img_b64 = test_image_cache[row['sclk']]
        
        current_parts = few_shot_parts.copy()
        current_parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        current_parts.append({"text": f"Sample ID (sclk): {row['sclk']} | k: {k_value}"})
        
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
    del test_image_cache
    gc.collect()
    
    return output_file

# Create Stage 2 self-correction refinement requests JSONL file
def create_self_correction_batch_input_file(reclass_df, explanations_pool, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "reclassify_requests_3p.jsonl")
    print(f"Generating Stage 2 self-correction request file: {output_file}...")
    
    class_explanations = {}
    for ex in explanations_pool:
        cls = ex['label']
        if cls not in class_explanations:
            class_explanations[cls] = []
        class_explanations[cls].append(ex)
        
    k_shot_examples = []
    for cls in ["3", "3-P"]:
        exs = class_explanations.get(cls, [])
        k_shot_examples.extend(exs[:4])
        
    used_sclk_ids = [ex['sclk'] for ex in k_shot_examples]
    
    few_shot_parts = []
    if k_shot_examples:
        few_shot_parts.append({"text": "Here are reference examples of Class 3 and Class 3-P spectra:"})
        for ex in k_shot_examples:
            label_key = ex['label']
            class_label = f"Class {label_key}"
            few_shot_parts.append({"text": f"Example: {class_label}\\\\n- Specific Sample Features: {ex['explanation']}"})
            few_shot_parts.append({"inline_data": {"mime_type": "image/png", "data": ex['image_base64']}})
            
    print(f"Pre-rendering target spectrum images for self-correction: {len(reclass_df)} samples...")
    test_image_cache = {}
    fig, ax = plt.subplots(figsize=(12, 6), dpi=60)
    
    for idx, row in reclass_df.iterrows():
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
            
    plt.close(fig)
    gc.collect()
    
    requests = []
    for _, row in reclass_df.iterrows():
        if row['sclk'] in used_sclk_ids:
            continue
            
        img_b64 = test_image_cache[row['sclk']]
        
        current_parts = [
            {"inline_data": {"mime_type": "image/png", "data": img_b64}}
        ]
        if few_shot_parts:
            current_parts.extend(few_shot_parts)
            
        exp_clean = row['explanation'].replace("{", "(").replace("}", ")")
        critique_prompt = CLASSIFICATION_USER_PROMPT_SELF_CORRECT.format(
            sclk=row['sclk'],
            first_pass_class=row['predicted_label'],
            first_pass_explanation=exp_clean
        )
        current_parts.append({"text": critique_prompt})
        
        request = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": current_parts
                    }
                ],
                "systemInstruction": {
                    "parts": [{"text": SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT}]
                },
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "STRING", "description": "The ID of the run (sclk)"},
                            "first_pass_analysis_critique": {"type": "STRING", "description": "Critique of the first-pass prediction"},
                            "class": {"type": "STRING", "description": "The final predicted class label, either '3' or '3-P'"},
                            "explanation": {"type": "STRING", "description": "Final explanation for the prediction"}
                        },
                        "required": ["id", "first_pass_analysis_critique", "class", "explanation"]
                    }
                }
            }
        }
        requests.append(json.dumps(request))
        
    with open(output_file, 'w') as f:
        for req in requests:
            f.write(req + '\n')
            
    print(f"Generated JSONL file with {len(requests)} requests at {output_file}")
    del test_image_cache
    gc.collect()
    
    return output_file

# 5. Submit and poll batch prediction job
def submit_and_poll_batch_job(configs, local_jsonl_path):
    project_id = configs.agent_settings.project_id
    bucket_name = configs.agent_settings.bucket_name
    
    filename = os.path.basename(local_jsonl_path)
    print(f"Uploading {local_jsonl_path} to GCS bucket: {bucket_name}")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
    
    gcs_input_path = f"input/{filename}"
    blob = bucket.blob(gcs_input_path)
    
    with open(local_jsonl_path, 'rb') as f:
        blob.chunk_size = 10 * 1024 * 1024
        blob.upload_from_file(f, content_type="application/json", timeout=600)
        
    gcs_source = f"gs://{bucket.name}/{gcs_input_path}"
    print(f"Uploaded {filename} to {gcs_source}")
    
    access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
    job_display_name = f"cda-3p-ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
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
    
    req_file = f"batch_request_3p_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        print(f"Error submitting job: {result.stderr}\\nResponse: {result.stdout}")
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

# 6. Stream and Parse batch results
def download_and_parse_batch_results_streaming(configs, job_status_data, test_df, output_parquet_path):
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
        
    print(f"Found {len(jsonl_blobs)} prediction JSONL files. Streaming and parsing...")
    
    parsed_data = []
    for blob in jsonl_blobs:
        print(f"  Streaming {blob.name} (Size: {blob.size / 1024 / 1024:.2f} MB) directly from GCS...")
        with blob.open("rt") as f:
            line_count = 0
            for line in f:
                if not line.strip():
                    continue
                parsed = extract_prediction_from_line(line)
                if parsed is None:
                    continue
                
                pred_id = parsed.get("id")
                pred_k = parsed.get("k")
                pred_class = parsed.get("class") or parsed.get("class_label") or "Noise"
                pred_explanation = parsed.get("explanation") or ""
                
                parsed_data.append({
                    "sclk": str(pred_id),
                    "predicted_label": pred_class,
                    "explanation": pred_explanation,
                    "k": int(pred_k) if pred_k is not None else 4
                })
                line_count += 1
                if line_count % 100 == 0:
                    print(f"    Streamed and parsed {line_count} predictions...")
                    
    predictions_df = pd.DataFrame(parsed_data)
    test_lookup = test_df[['sclk', 'class']].copy()
    test_lookup['sclk'] = test_lookup['sclk'].astype(str)
    
    merged_df = pd.merge(predictions_df, test_lookup, on='sclk', how='inner')
    merged_df.rename(columns={"class": "true_label"}, inplace=True)
    merged_df = merged_df[['sclk', 'true_label', 'predicted_label', 'explanation', 'k']]
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    merged_df.to_parquet(output_parquet_path)
    print(f"Saved results to {output_parquet_path}")
    return merged_df

# 7. Evaluate and Save
def generate_metrics_and_plots(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    k_values = sorted(results_df['k'].unique())
    metrics_summary = {}
    
    print("\n" + "="*50 + "\n3-P focused Ablation Study Summary\n" + "="*50)
    
    for k in k_values:
        sub_df = results_df[results_df['k'] == k]
        
        # Overall accuracy
        overall_acc = accuracy_score(sub_df['true_label'], sub_df['predicted_label'])
        print(f"\nk = {k}-shot classification:")
        print(f"  Overall Accuracy: {overall_acc:.2%}")
        
        # Calculate confusion matrix for all classes
        labels = sorted(results_df['true_label'].unique())
        cm = confusion_matrix(sub_df['true_label'], sub_df['predicted_label'], labels=labels)
        
        # Save Confusion Matrix as CSV
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_csv_path = os.path.join(output_dir, f"confusion_matrix_k{k}.csv")
        cm_df.to_csv(cm_csv_path)
        print(f"  Saved confusion matrix CSV to {cm_csv_path}")
        
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix (k={k} shots) - Acc: {overall_acc:.2%}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, f"confusion_matrix_k{k}.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"  Saved confusion matrix plot to {cm_path}")
        
        # Calculate class-specific metrics for 3-P
        target_cls = "3-P"
        
        # True Positives, False Positives, False Negatives, True Negatives
        y_true_binary = (sub_df['true_label'] == target_cls).astype(int)
        y_pred_binary = (sub_df['predicted_label'] == target_cls).astype(int)
        
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
        if cm_binary.shape == (2, 2):
            tn, fp, fn, tp = cm_binary.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            if len(y_true_binary.unique()) == 1:
                val = y_true_binary.iloc[0]
                if val == 0:
                    tn = len(y_true_binary)
                else:
                    tp = len(y_true_binary)
            
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        print(f"  Class 3-P Specific Metrics:")
        print(f"    TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"    Precision: {precision:.2%}")
        print(f"    Recall (Sensitivity): {recall:.2%}")
        print(f"    F1-score: {f1:.2%}")
        
        metrics_summary[str(k)] = {
            "overall_accuracy": float(overall_acc),
            "class_3p_metrics": {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        }
        
    metrics_json_path = os.path.join(output_dir, "metrics_3p_ablation_self_correct.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nSaved metrics summary JSON to {metrics_json_path}")

async def main():
    configs = Config()
    client = genai.Client()
    model_id = configs.agent_settings.model
    
    # 1. Fetch, Parse, QI Filter, Scale, Smooth
    df = download_and_preprocess_data()
    
    # 2. Split train/test (With test_size_per_class=100 cap per class)
    train_df, test_df, train_sclks = split_train_test(
        df, 
        max_train_map={'3-P': 4, '5': 16, '5-Na': 16},
        test_size_per_class=100
    )
    
    # 3. Generate VLM explanations pool
    explanations_pool = await generate_explanations_pool(client, model_id, train_df, train_sclks)
    
    # 4. Generate Single Request JSONL File (Stage 1)
    k_value = 4
    output_dir = os.path.abspath(os.path.join(project_root, "Notebooks/study_results_3p_qi_filtered"))
    batch_file = create_ablation_batch_input_file(test_df, explanations_pool, k_value, output_dir)
    
    # 5. Submit and Poll Batch Job (Stage 1 - Sequential Single Job)
    print("\n--- STAGE 1: Multiclass Batch Predictions (Single Job) ---")
    job_status = submit_and_poll_batch_job(configs, batch_file)
    
    # 6. Stream download and parse results
    print("\nDownloading and parsing Stage 1 prediction results (streaming)...")
    output_parquet_stage1 = os.path.join(output_dir, "results_3p_ablation_stage1.parquet")
    stage1_df = download_and_parse_batch_results_streaming(configs, job_status, test_df, output_parquet_stage1)
    
    # 7. Identify samples predicted as "3" or "3-P" for Stage 2 Self-Correction
    reclass_candidates = stage1_df[stage1_df['predicted_label'].isin(["3", "3-P"])].copy()
    print(f"\nFound {len(reclass_candidates)} candidates predicted as '3' or '3-P' for Stage 2 self-correction.")
    
    if len(reclass_candidates) > 0:
        # Get corresponding rows with full spectrum data from test_df
        reclass_test_df = test_df[test_df['sclk'].astype(str).isin(reclass_candidates['sclk'].astype(str))].copy()
        
        # Merge Stage 1 predicted_label and explanation columns into reclass_test_df
        candidate_info = reclass_candidates[['sclk', 'predicted_label', 'explanation']].copy()
        candidate_info['sclk'] = candidate_info['sclk'].astype(str)
        reclass_test_df['sclk'] = reclass_test_df['sclk'].astype(str)
        reclass_test_df = pd.merge(reclass_test_df, candidate_info, on='sclk', how='inner')
        
        # Generate Stage 2 Requests JSONL file
        reclass_batch_file = create_self_correction_batch_input_file(reclass_test_df, explanations_pool, output_dir)
        
        # Submit and poll Stage 2 job
        print("\n--- STAGE 2: VLM Self-Correction Predictions (Single Job) ---")
        job_status_reclass = submit_and_poll_batch_job(configs, reclass_batch_file)
        
        # Download and parse Stage 2 results
        print("\nDownloading and parsing Stage 2 prediction results (streaming)...")
        output_parquet_stage2 = os.path.join(output_dir, "results_3p_ablation_stage2.parquet")
        stage2_df = download_and_parse_batch_results_streaming(configs, job_status_reclass, reclass_test_df, output_parquet_stage2)
        
        # Merge refinement results back into predictions
        print("\nMerging refinement predictions...")
        refinement_map = dict(zip(stage2_df['sclk'].astype(str), stage2_df['predicted_label']))
        
        final_preds = []
        for _, row in stage1_df.iterrows():
            sclk_str = str(row['sclk'])
            pred = row['predicted_label']
            if sclk_str in refinement_map:
                refined_pred = refinement_map[sclk_str]
                print(f"  SCLK {sclk_str}: Refined prediction from {pred} -> {refined_pred}")
                pred = refined_pred
            final_preds.append(pred)
            
        stage1_df['predicted_label'] = final_preds
        
    # Save Final Integrated Results
    output_parquet_final = os.path.join(output_dir, "results_3p_ablation.parquet")
    stage1_df.to_parquet(output_parquet_final)
    print(f"Saved integrated hierarchical classification results to {output_parquet_final}")
    
    # Save as CSV with sclk, true_label, predicted_label, explanation
    output_csv_final = os.path.join(output_dir, "results_3p_ablation.csv")
    stage1_df[['sclk', 'true_label', 'predicted_label', 'explanation']].to_csv(output_csv_final, index=False)
    print(f"Saved results CSV to {output_csv_final}")
    
    # 8. Generate Metrics & Plots for the Integrated Predictions
    generate_metrics_and_plots(stage1_df, output_dir)
    print("3-P Hierarchical Ablation Study (Self-Correction Loop) with QI filtering completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
