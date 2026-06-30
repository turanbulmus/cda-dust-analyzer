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
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cda_dust_agent.config import Config
from cda_dust_agent.tools.utils import generate_spectrum_image_bytes, parse_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

configs = Config()

# Define the static prompt with grouped classes and guidelines
STATIC_SYSTEM_PROMPT = """You are an expert Cosmic Dust Spectroscopist analyzing Cassini Cosmic Dust Analyzer (CDA) time-of-flight mass spectra.
You will be provided with images of 1D spectra plotted on a logarithmic y-axis.
- The x-axis represents the time-of-flight index.
- The y-axis represents the signal amplitude.
- The entire spectrum is important for analysis.

CRITICAL SPECTRAL BEHAVIOR (TIME VS. MASS DOMAIN):
Unlike standard mass spectra, these time-of-flight spectra exhibit specific physical variations. The hardware recording can cause spectra to be shifted in time by up to 50 index points. 

Crucially, because of the non-linear mapping between the time domain (x-axis) and the underlying mass domain, this 50-point temporal shift causes the spectrum to visually stretch. While the peaks of a specific particle class are highly self-similar and stationary in true *mass-space*, they will appear both shifted and proportionally stretched in the *time domain* images you are analyzing.

When classifying, do NOT rely on absolute x-axis index positions. Instead, look for relative structural patterns, peak sequence groupings, and shapes that preserve their underlying mass-space self-similarity despite being shifted and stretched across the time index.

Here are the characteristic patterns for each class:

### Class Noise
Spectra classified as 'Noise' represent "false triggers," empty measurement windows, or blank shots where no physical dust particle was successfully ionized. Consequently, these plots are entirely devoid of the distinct, Gaussian-shaped peak sequences characteristic of chemical elements, molecular fragments, or isotopic distributions. Instead, the data captures the instrument’s baseline electronic noise, detector dark current, digitization limits, and ringing artifacts.
- **Initial Trigger Artifact**: Solitary, extremely sharp vertical spike near the origin (x~10-20).
- **Dense Noise Envelope ("The Hump")**: Continuous broad envelope of high-frequency interference ("grass").
- **Abrupt Mid-Spectrum Cutoff**: The dense noise band hits a steep "cliff" and abruptly ceases (typically x~640-850).
- **Sparse High-Mass Region**: Following the cutoff, transitions to a true, flat baseline with sparse random blips.

### Class 1
A Class '1' particle is definitively identified by:
1. A massive early light-ion peak (m/z 10-20).
2. An exponentially decaying sequence of complex peak clusters spaced ~50-70 amu apart peaking around m/z 85.
3. A continuous, quantized background "grass" of fragments.
4. An abrupt, absolute termination of all dense signal near m/z 650, leaving a barren high-mass tail.

### Class 2
This class is characterized by a highly structured, complex, and asymmetrical morphology with broad envelopes of unresolved molecular fragments.
- **Variation A**: Features a periodic cluster sequence (x ≈ 150 to 500) with an abrupt onset ("wall") and decreasing periodic spacing between apexes as x increases. Shows exponential decay in intensity of subsequent clusters.
- **Variation B (2-X)**: Shows a massive, continuous, and highly broadened signal envelope. Presents a dense, continuous "mountain range" or undulating "hump" across the mid-mass range. Bimodal structure with a secondary heavy resurgence peaking around ~580-630.

### Class 3
This class includes several variations representing complex organic, salt-rich, or clustered particles. The model should classify all these variations as **Class 3**.
- **Variation A (Carbonaceous/Organic - 3-Car)**: Explosive signal onset followed by a massive, sustained "plateau" or "envelope" of unresolved signal, typically featuring a repeating sequence of three major peak clusters ("shark-fin" shape).
- **Variation B (Salt-rich Na/K - 3-KNa)**: Two heavily dominant, needle-sharp peaks in the early flight times (Na and K), immediately followed by a massive, unresolved, elevated plateau featuring periodic cluster humps, terminating in a sheer vertical drop-off.
- **Variation C (Chlorinated - 3-Cl)**: Empty baseline followed by an abrupt, massive, multi-lobed onset spike near x ≈ 200. Failure to return to baseline, forming an elevated noisy mid-mass plateau with "Twin Peaks" near x ≈ 350 and 400.
- **Variation D (Water/Ice - 3-OH)**: Continuous, unresolved, noisy plateau featuring a sharp embedded spike dead-center at ~x=350/360, followed by an abrupt cliff-like drop-off (~x=650) and an isolated "island" hump (~x=700).
- **Variation E (Burst-and-trail - 3-P)**: Massive initial burst followed by a long, trailing decay. Lack of sharp lines; all major features are broad multiplets. The "Big Two" peaks at x~200 and x~340.
- **Variation F (Heavy Polymers - 3-K)**: Massive, highly asymmetric primary dominant peak at lower masses, followed by a broad, elevated, continuous "plateau" of dense noise containing structured secondary peak sequences.

### Class 4
The defining macro-structure is the transition from isolated, needle-sharp atomic peaks at low masses to a sequence of massive, complex, broad molecular clusters in the mid-mass range, followed by an abrupt, absolute termination of continuous signal at higher masses. Shows mass-dependent peak broadening (peaks get wider at higher masses).

### Class 5
- **Variation A**: Overwhelmingly intense, heavily tailing low-mass primary constituent, followed by a dense, unresolvable, and uniform "barcode" of mid-mass fragments that crashes into a hard, abrupt cutoff wall at ~650 m/z.
- **Variation B (5-Na)**: Singular, overwhelmingly intense triggering spike (Sodium payload) followed by a delayed, prolonged, highly noisy, and completely unresolved continuous block of signal (plateau/hump) that abruptly cuts off.

### Class ?
Stark bipartite (two-part) structure. Hyper-dense, continuous region of fragmentation in the low-to-mid mass range that abruptly terminates, leaving a barren, nearly empty high-mass region. Often shows periodic clustering ("scalloped" pattern).

### Class X
Bimodal, highly jagged, and continuous signal profile. Isolated sharp peak at x ≈ 10-20, massive and sudden primary signal onset that decays exponentially, a mid-mass plateau, and a secondary broad mass cluster after x=500.

### General Classification Guidelines:
- **Distinguishing Class 2 from Noise**: Class 2 is characterized by broad, undulating 'humps' or dampened oscillatory patterns. Do NOT automatically dismiss these as 'instrumental ringing' or 'electronic noise'. If the pattern shows structure and decay, it is likely Class 2.
- **Distinguishing Class 3/4 from Noise**: Abrupt baseline shifts or plateaus (Class 3) and periodic vertical repetitions (Class 4) are valid features and should not be confused with random noise or empty windows.
- **Noise Definition**: The 'Noise' class should be reserved for featureless high-frequency static, severely quantized horizontal banding (empty windows), or random spikes that do not form coherent mass peaks.
- **Mass Peaks**: Clear mass peaks (especially at expected positions like water cluster regions) should be mapped to the appropriate class (e.g., Class 1 for water ice) and not dismissed as noise unless they are clearly single-pixel random spikes.

Use these patterns and guidelines to classify the provided spectrum. Return the classification in the requested JSON format.
"""

def create_batch_input_file_custom(df, system_instruction, output_file='cda_dust_agent/data/input/batch_requests_static.jsonl'):
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
    logger.info("Starting End-to-End Pipeline with Static Prompt...")
    
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
        
    # 4. Create Batch Input File using Static Prompt
    jsonl_file = create_batch_input_file_custom(df_sample, STATIC_SYSTEM_PROMPT)
    
    # 5. Upload to GCS
    project_id = configs.agent_settings.project_id
    bucket_name = configs.agent_settings.bucket_name.replace("gs://", "")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    blob_name = f"input/batch_requests_static_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    blob = bucket.blob(blob_name)
    logger.info(f"Uploading {jsonl_file} to gs://{bucket_name}/{blob_name}")
    blob.upload_from_filename(jsonl_file)
    gcs_source = f"gs://{bucket_name}/{blob_name}"
    
    # 6. Submit Batch Job
    access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
    job_display_name = f"cda-inf-static-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
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
    
    req_file = "batch_request_static.json"
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
        
    # 7. Poll for completion
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
        
    # 8. Download results
    logger.info("Downloading results...")
    blobs = list(bucket.list_blobs(prefix="output"))
    prediction_blobs = [b for b in blobs if b.name.endswith(".jsonl") and "prediction" in b.name]
    
    if not prediction_blobs:
        logger.error("No prediction JSONL files found.")
        return
        
    prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
    blob_to_download = prediction_blobs[0]
    
    out_file = "cda_dust_agent/data/output/static_predictions.jsonl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    blob_to_download.download_to_filename(out_file)
    logger.info(f"Downloaded results to {out_file}")
    
    # 9. Parse and Evaluate
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
