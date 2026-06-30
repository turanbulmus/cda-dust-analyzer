import os
from dotenv import load_dotenv
load_dotenv()
import json
import base64
import numpy as np
import pandas as pd
import asyncio
from google import genai
from google.genai import types
import logging
import subprocess
import time
from datetime import datetime
from google.cloud import storage
import vertexai
from vertexai import generative_models
from vertexai import types as vx_types
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cda_dust_agent.config import Config
from cda_dust_agent.tools.utils import generate_spectrum_image_bytes, parse_response
from cda_dust_agent.tools.prompt_refinery import get_spectrum_description

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

configs = Config()

# Modified prompt refinery that takes a dataframe
async def run_prompt_refinery_consolidated(df, client, vx_client, project_id, max_levels=3):
    logger.info("Starting consolidated prompt refinery...")
    
    # Stratified sampling of 30 observations per class
    classes = df['class'].unique()
    sampled_dfs = []
    for cls in classes:
        cls_df = df[df['class'] == cls]
        n_samples = min(30, len(cls_df))
        if n_samples > 0:
            sampled_dfs.append(cls_df.sample(n=n_samples, random_state=42))
            
    sample_df = pd.concat(sampled_dfs).reset_index(drop=True)
    logger.info(f"Sampled {len(sample_df)} observations across {len(classes)} classes for prompt refinement.")
    
    # Level 1: Generate descriptions
    logger.info("Level 1: Generating descriptions for each sample...")
    semaphore = asyncio.Semaphore(5)
    
    async def process_row(row):
        async with semaphore:
            desc = await get_spectrum_description(client, np.array(row['spectrum']), row['class'], str(row['sclk']))
            return {
                "sclk": row['sclk'],
                "class": row['class'],
                "description": desc
            }
            
    tasks = [process_row(row) for _, row in sample_df.iterrows()]
    level_1_results = await asyncio.gather(*tasks)
    
    # Save Level 1 results
    os.makedirs("cda_dust_agent/data/results", exist_ok=True)
    pd.DataFrame(level_1_results).to_csv("cda_dust_agent/data/results/prompt_study_level_1_consolidated.csv", index=False)
    
    current_descriptions = {}
    for res in level_1_results:
        cls = res['class']
        desc = res['description']
        if cls not in current_descriptions:
            current_descriptions[cls] = []
        current_descriptions[cls].append(desc)
        
    level_num = 2
    active_classes = list(current_descriptions.keys())
    final_patterns = {}
    
    while active_classes and level_num <= max_levels:
        logger.info(f"Level {level_num}: Consolidating patterns for {len(active_classes)} classes...")
        next_descriptions = {}
        new_active_classes = []
        
        async def process_class_autonomous(cls, descs):
            prompt = (
                f"Here are the descriptions of spectra for Class '{cls}'.\n"
                f"1. Identify how many distinct characteristic patterns are present in these descriptions.\n"
                f"2. Provide a detailed description for each distinct pattern.\n"
                f"3. Decide if these patterns can be consolidated further in a next step to reach a single unified pattern for this class, "
                f"or if they represent the fundamental distinct types for this class that should not be merged.\n"
                f"Return the result as a JSON object."
            )
            text_parts = [prompt]
            for i, d in enumerate(descs):
                text_parts.append(f"Description {i+1}:\n{d}\n")
            prompt_text = "\n".join(text_parts)
            
            try:
                response = await client.aio.models.generate_content(
                    model=configs.agent_settings.model,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "patterns": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                                "can_consolidate_further": types.Schema(type=types.Type.BOOLEAN),
                                "explanation": types.Schema(type=types.Type.STRING)
                            },
                            required=["patterns", "can_consolidate_further", "explanation"]
                        )
                    )
                )
                return cls, json.loads(response.text)
            except Exception as e:
                logger.error(f"Failed at Level {level_num} for Class {cls}: {e}")
                return cls, {"patterns": [f"Failed to consolidate: {e}"], "can_consolidate_further": False, "explanation": str(e)}

        tasks = [process_class_autonomous(cls, current_descriptions[cls]) for cls in active_classes]
        level_results = await asyncio.gather(*tasks)
        
        for cls, result_data in level_results:
            patterns = result_data.get("patterns", [])
            can_consolidate = result_data.get("can_consolidate_further", False)
            
            if can_consolidate and len(patterns) > 1:
                next_descriptions[cls] = patterns
                new_active_classes.append(cls)
            else:
                final_patterns[cls] = patterns
                
        current_descriptions = next_descriptions
        active_classes = new_active_classes
        level_num += 1
        
    for cls in current_descriptions:
        if cls not in final_patterns:
            final_patterns[cls] = current_descriptions[cls]
            
    # Generate Final Prompt
    system_prompt = "You are an expert system for classifying Cassini Cosmic Dust Analyzer (CDA) Time-of-Flight mass spectra.\n"
    system_prompt += "Here are the characteristic patterns for each class identified through autonomous hierarchical analysis:\n\n"
    
    for cls, patterns in final_patterns.items():
        system_prompt += f"### Class {cls}\n"
        for i, pattern in enumerate(patterns):
            system_prompt += f"Pattern {i+1}:\n{pattern}\n\n" if len(patterns) > 1 else f"{pattern}\n\n"
            
    system_prompt += "\n### General Classification Guidelines:\n"
    system_prompt += "- **Distinguishing Class 2 from Noise**: Class 2 is characterized by broad, undulating 'humps' or dampened oscillatory patterns. Do NOT automatically dismiss these as 'instrumental ringing' or 'electronic noise'. If the pattern shows structure and decay, it is likely Class 2.\n"
    system_prompt += "- **Distinguishing Class 3/4 from Noise**: Abrupt baseline shifts or plateaus (Class 3) and periodic vertical repetitions (Class 4) are valid features and should not be confused with random noise or empty windows.\n"
    system_prompt += "- **Noise Definition**: The 'Noise' class should be reserved for featureless high-frequency static, severely quantized horizontal banding (empty windows), or random spikes that do not form coherent mass peaks.\n"
    system_prompt += "- **Mass Peaks**: Clear mass peaks (especially at expected positions like water cluster regions) should be mapped to the appropriate class (e.g., Class 1 for water ice) and not dismissed as noise unless they are clearly single-pixel random spikes.\n\n"
    
    system_prompt += "Use these patterns and guidelines to classify the provided spectrum. Return the classification in the requested JSON format.\n"
    
    return system_prompt

# Modified create_batch_input_file to accept custom system instruction
def create_batch_input_file_custom(df, system_instruction, output_file='cda_dust_agent/data/input/batch_requests_consolidated.jsonl'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"Generating batch input file: {output_file}...")
    
    from cda_dust_agent.prompts import CLASSIFICATION_USER_PROMPT
    
    requests = []
    for _, row in df.iterrows():
        spect = row['spectrum']
        img_bytes = generate_spectrum_image_bytes(spect, title=f"Sample {row['sclk']}")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        parts = [
            {"text": CLASSIFICATION_USER_PROMPT},
            {"inline_data": {"mime_type": "image/png", "data": img_b64}},
            {"text": f"Sample ID (sclk): {row['sclk']}"}
        ]
        
        request = {
            "request": {
                "contents": [{"role": "user", "parts": parts}],
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "STRING"},
                            "class": {"type": "STRING"},
                            "explanation": {"type": "STRING"}
                        },
                        "required": ["id", "class", "explanation"]
                    }
                }
            }
        }
        requests.append(json.dumps(request))
        
    with open(output_file, 'w') as f:
        for req in requests:
            f.write(req + '\n')
            
    return output_file

async def main():
    logger.info("Starting End-to-End Pipeline with Consolidated Labels...")
    
    # 1. Load Data
    train_file = "cda_dust_agent/data/raw/cda_train.parquet"
    if not os.path.exists(train_file):
        logger.error(f"Training file not found: {train_file}")
        return
        
    df = pd.read_parquet(train_file)
    
    # 2. Consolidate Labels
    logger.info("Consolidating labels...")
    df['class'] = df['class'].apply(lambda x: str(x).split('-')[0] if pd.notna(x) else 'Noise')
    
    # 3. Sample 2000 rows
    logger.info("Sampling 2000 rows randomly...")
    if len(df) > 2000:
        df_sample = df.sample(n=2000, random_state=42).reset_index(drop=True)
    else:
        df_sample = df
        logger.warning(f"Dataset only has {len(df)} rows, using all.")
        
    # 4. Initialize Clients
    project_id = configs.agent_settings.project_id
    client = genai.Client(vertexai=True, project=project_id, location="global")
    vx_client = vertexai.Client(project=project_id, location="us-central1")
    
    # 5. Run Prompt Refinery (using the 2000 samples)
    refined_prompt = await run_prompt_refinery_consolidated(df_sample, client, vx_client, project_id, max_levels=2)
    logger.info("Refined prompt generated.")
    
    # Save refined prompt
    with open("cda_dust_agent/data/prompt_optimizer/refined_prompt_consolidated.txt", "w") as f:
        f.write(refined_prompt)
        
    # 6. Create Batch Input File
    jsonl_file = create_batch_input_file_custom(df_sample, refined_prompt)
    
    # 7. Upload to GCS
    bucket_name = configs.agent_settings.bucket_name.replace("gs://", "")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    blob_name = f"input/batch_requests_consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    blob = bucket.blob(blob_name)
    logger.info(f"Uploading {jsonl_file} to gs://{bucket_name}/{blob_name}")
    blob.upload_from_filename(jsonl_file)
    gcs_source = f"gs://{bucket_name}/{blob_name}"
    
    # 8. Submit Batch Job
    access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
    job_display_name = f"cda-inf-consolidated-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
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
    
    req_file = "batch_request_consolidated.json"
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
        
    # 9. Poll for completion
    check_url = f"https://aiplatform.googleapis.com/v1/{job_name}"
    logger.info("Polling job status...")
    
    while True:
        access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
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
            logger.error(f"Job Ended with state: {state}")
            return
            
        await asyncio.sleep(30)
        
    # 10. Download results
    logger.info("Downloading results...")
    blobs = list(bucket.list_blobs(prefix="output"))
    prediction_blobs = [b for b in blobs if b.name.endswith(".jsonl") and "prediction" in b.name]
    
    if not prediction_blobs:
        logger.error("No prediction JSONL files found.")
        return
        
    prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
    blob_to_download = prediction_blobs[0]
    
    out_file = "cda_dust_agent/data/output/consolidated_predictions.jsonl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    blob_to_download.download_to_filename(out_file)
    logger.info(f"Downloaded results to {out_file}")
    
    # 11. Parse and Evaluate
    logger.info("Parsing results and running analysis...")
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
        
        # Consolidate predicted label as well just in case
        pred_label = str(raw_label).split('-')[0].strip()
        
        pred_id = parsed.get("id")
        if pred_id is not None:
            pred_id_str = str(pred_id)
            if pred_id_str in truth_map:
                y_true.append(truth_map[pred_id_str])
                y_pred.append(pred_label)
                
    logger.info(f"Matched {len(y_true)} predictions.")
    
    target_names = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=target_names)
    logger.info("\n" + report)

if __name__ == "__main__":
    asyncio.run(main())
