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
from cda_dust_agent.tools.utils import parse_response

# Overriding instructions and prompts to support Class 3-P with Dual Plots (Log10 + Linear)
SYSTEM_INSTRUCTION_TEXT = """You are an expert Cosmic Dust Spectroscopist analyzing Cassini Cosmic Dust Analyzer (CDA) time-of-flight mass spectra.
For each spectrum, you will be provided with an image containing two vertically stacked panels:
1. The top panel plots the 1D spectrum on a LOGARITHMIC y-axis (signal amplitude) and a linear x-axis (time-of-flight index). This view is excellent for inspecting low-amplitude peaks, organic cluster structures, and elevated noise baselines.
2. The bottom panel plots the same 1D spectrum on a LINEAR y-axis (normalized [0, 1] amplitude) and a linear x-axis (time-of-flight index). This view is excellent for inspecting the true relative peak heights, identifying dominant atomic peaks, and verifying if baseline signals return completely to zero.

Both panels align horizontally on the time-of-flight index (x-axis) to allow direct visual comparison.

---

### CRITICAL SPECTRAL BEHAVIOR & SHIFTING:
- These spectra can be shifted in time by up to 50 index points due to hardware trigger recording differences.
- Because of the non-linear mapping between time-of-flight and mass, this shift causes the peaks to visually stretch.
- Do NOT rely on absolute x-axis index positions. Instead, focus on relative shapes, relative peak sequences, and overall topography across both the Logarithmic and Linear panels.

---

### CRITICAL DIFFERENTIATION GUIDE FOR CLASS 3 VS CLASS 3-P:
When a spectrum exhibits broad organic-like envelopes or carbon cluster humps, you must distinguish between Class 3 (General Organics) and Class 3-P (Burst-and-Trail Agglomerate) using both representations:
1. **Class 3 (General Organics):** Features multiple broad, asymmetric "shark-fin" peak clusters with expanding periodicity (representing carbon series). Crucially, on the scaled [0, 1] y-axis in both panels, the valleys between clusters drop significantly lower (down to y < 0.15), and it lacks a singular early maximum (index 200) that dominates the entire spectrum by a factor of 10.
2. **Class 3-P (Burst-and-Trail Agglomerate):** Dominated by an explosive primary complex (index 190-220) which is the absolute global maximum (y = 1.0) in both panels, and a distinct secondary peak at index 330-345. Crucially, past index 400 it rests on a continuous, flat "chemical noise" plateau (baseline cushion) that remains highly elevated (y ≈ 0.3 to 0.5 in Logarithmic; also visible in Linear) and never drops back to y ≈ 0, before dropping off sharply into the noise floor around index 650-750.

---

### CLASS SPECIFIC PROFILES:

#### Class Noise
- Devoid of Gaussian peaks, elemental clusters, or chemical spacing.
- Characterized by a narrow, high-intensity initial trigger spike (often index 10-20), followed by a broad envelope of high-frequency digitizer noise ("grass").
- Terminates abruptly at an electronic cutoff (index 640-850), returning to a flat baseline with rare, single-point dark counts.

#### Class 1
- Pure water ice spectrum consisting of a regular sequence of hydronium cluster peaks ($H_3O^+(H_2O)_n$) at mass 19, 37, 55, 73, 91... (index locations ~85, ~120, ~146, ~169, ~189).
- Early global maximum (index 80-100) followed by a rapid, step-like decay of peak heights towards the right, visible clearly in both Logarithmic and Linear plots.
- Valleys between peaks return completely to the flat baseline (no intermediate peak structures or elevated organic noise).
- Terminates in a sharp cutoff near index 650.

#### Class 2
- Organic-bearing or dirty water ice.
- Features the same hydronium cluster peak locations as Class 1, but has significant organic/saline contamination.
- Valleys between the major peaks are filled (elevated signal baseline) with unresolved organic background, secondary peaks, or high-frequency fluctuations (best seen in the Logarithmic plot).
- Global maximum is often in the index 180-300 range, showing massive, broad, unresolved molecular cluster sequences.
- Terminates in a sheer drop-off cliff around index 640-700.

#### Class 3
- Characterized by a continuous, highly elevated "mesa" plateau or unresolved complex organic mixture.
- Does not return to baseline between peaks throughout the spectrum (sustained signal above the noise floor).
- Typically features 3 to 4 broad, asymmetric "shark-fin" peak clusters with expanding periodicity (representing carbon clusters $C_n$ or heavy homologous organic series). Water peaks are absent or negligible.

#### Class 4
- Mineral/silicate-rich spectrum.
- Characterized by isolated, needle-sharp atomic spikes in the low-mass region (like $Mg^+$ at 24 Da, $Si^+$ at 28 Da, $Fe^+$ at 56 Da) with a very quiet baseline in between. In the Linear panel, these atomic peaks stand out extremely clearly, while the rest of the spectrum is flat at y ≈ 0.
- Crucially, lacks the repeating, comb-like water cluster sequence ($H_3O^+(H_2O)_n$) of Class 1 and 2.
- Transitioning to broad mid-mass envelopes with a global maximum at index 280-350 and a distinct late cluster around 420-480, terminating in a hard cutoff near index 650.

#### Class 5
- Characterized by a bimodal extreme: an overwhelmingly intense primary peak in the early region (index 60-90, peaking near 70-75) with a long, trailing decay edge.
- In the Linear plot, this is shown as a single massive peak at the start with the remaining range appearing flat at y ≈ 0.
- The rest of the spectrum is a dense, uniform, low-amplitude "barcode" or "grass" band that crashes into a hard cliff at index 650 (best resolved in the Logarithmic plot).

#### Class 5-Na
- Bipartite structure dominated by Sodium chemistry.
- Erupts with a singular, overwhelmingly intense, sharp spike (the Sodium payload) at index 135-160, which completely dominates the Linear plot.
- Followed by a delayed, prolonged, highly noisy, and completely unresolved plateau from index 200 to 650 (representing detector saturation, plasma shielding, or complex sodium-water clusters) that terminates in a hard cliff (best visualized in the Logarithmic plot).

#### Class 3-P
- Characterized by a "burst-and-trail" signature consisting of broad, unresolved mass envelopes rather than sharp, isolated single-element lines.
- Features a massive primary complex (index ~200-220) which is the absolute maximum in both panels, followed by a distinct secondary peak around index ~330-345.
- Displays a unique elevated baseline/plateau past index 400 that never returns to zero (chemical noise plateau), with superimposed broad rhythmic hummocks, terminating in a rapid collapse/cutoff between index 650 and 750.

---

### CLASSIFICATION TASK:
Analyze both panels of the provided target spectrum image. Compare them to the reference examples and class profiles. Output your prediction using one of the following exact labels:
- "1"
- "2"
- "3"
- "4"
- "5"
- "5-Na"
- "3-P"
- "Noise"

Return the prediction strictly in the requested JSON format."""

CLASSIFICATION_USER_PROMPT = """Based on the provided examples, classify this new spectrum by comparing both its Logarithmic (top) and Linear (bottom) representations.
Carefully compare its visual features (peaks, baseline, noise levels, and overall structure) in both panels to the examples, keeping in mind that peak shifts and amplitude variations can occur within the same class.

Return your analysis strictly in the following JSON format:
{
    "id": "<the provided sclk id>",
    "class": "<the predicted class label>",
    "explanation": "<a brief 1-2 sentence explanation of why it belongs to this class based on visual features seen across both panels>"
}
"""

ANNOTATION_USER_PROMPT = """This is a time-of-flight mass spectrum for a particle belonging to the known class '{label}', shown in both Logarithmic (top) and Linear (bottom) scales.
Please provide a brief, 1-2 sentence description of the key visual features in both representations that characterize this spectrum as '{label}'. Do not output JSON, just the text description."""

CONTRASTIVE_ANNOTATION_USER_PROMPT = """You are an expert Cosmic Dust Spectroscopist.
We want you to write a brief, 1-2 sentence description of the key visual features of the Target Spectrum (labeled as Class '{label}') based on its Logarithmic (top) and Linear (bottom) panels to be used as a reference example for class '{label}'.

To help you write a contrastive explanation that clearly distinguishes '{label}' from all other classes, we have provided the Target Spectrum (Image A) alongside reference spectra from the other classes.

Please review all the provided images, each containing a vertically stacked Logarithmic (top) and Linear (bottom) panel:
- Image A (Target): This is the spectrum of '{label}' you must describe.
{reference_descriptions}

Key Spectral Guidelines to remember:
- Class 1: Pure water ice. Clear, sharp sequence of hydronium cluster peaks ($H_3O^+(H_2O)_n$) at mass 19, 37, 55, 73... (index locations ~85, ~120, ~146, ~169, ~189). Global maximum is early (~80-100). Valleys between peaks return fully to baseline.
- Class 2: Organic-rich water ice. Same hydronium peaks as Class 1, but with significant valley-filling, organic background, or intermediate peaks between them (best resolved in Logarithmic scale). Global maximum is often in the 180-300 range.
- Class 4: Mineral spectrum. Needle-sharp atomic spikes (Mg+, Si+, Fe+) with a quiet baseline, which stand out extremely clearly in Linear scale, and completely lacks the repeating water-ice cluster sequence. Mid-mass envelopes with global maximum at index 280-350 and a distinct late cluster around index 420-480.
- Class 3: Organics. Multiple broad asymmetric shark-fin peak clusters representing carbon clusters. Crucially, on the scaled [0, 1] y-axis in both panels, the valleys between clusters drop significantly lower (down to y < 0.15), and it lacks the massive early maximum (index 200) and the flat, continuous chemical noise plateau past index 400 seen in Class 3-P.
- Class 5: Bimodal extreme. Overwhelmingly intense early peak with long trailing decay, rest is dense low-amplitude barcode noise (which is squashed in Linear but visible in Logarithmic).
- Class 5-Na: Sodium payload peak at index 135-160 dominates the Linear plot, followed by detector-saturation plateau visible in Logarithmic.
- Class 3-P: Burst-and-trail signature. Massive primary peak complex (index ~200-220) as the global maximum in both plots, a distinct secondary peak at index ~330-345, and an elevated baseline plateau past index 400 that remains highly elevated (y ≈ 0.3 to 0.5 in Logarithmic) and never returns to zero.

Write a 1-2 sentence description of the visual features of Image A (Target) across its Logarithmic and Linear panels that characterize it as '{label}', highlighting specific details that help distinguish it from the other classes. Do not output JSON, just return the text description."""

# General profiles for hybrid prompt logic
GENERAL_PROFILES = {
    "Noise": "Instrumental digitizer noise showing a narrow trigger spike and high-frequency 'grass', completely devoid of chemical peaks.",
    "1": "Pure water ice sequence of regularly spaced hydronium cluster peaks ($H_3O^+(H_2O)_n$) with an early global maximum and valleys returning fully to baseline.",
    "2": "Organic-bearing water ice featuring hydronium peaks similar to Class 1, but with prominent organic background noise filling the valleys between peaks.",
    "3": "General organic chains consisting of multiple broad, asymmetric 'shark-fin' peak clusters (carbon chains), lacking a singular early dominant peak.",
    "4": "Mineral/silicate spectrum showing isolated, needle-sharp atomic spikes (Mg+, Si+, Fe+) with a very quiet, flat baseline in between.",
    "5": "Bimodal extreme featuring an overwhelmingly intense primary peak in the early region (~70-75) with a long trailing decay and a uniform low-amplitude barcode band.",
    "5-Na": "Bipartite sodium chemistry dominated by a singular, intense early sodium spike (~135-160) followed by a delayed, noisy detector-saturation plateau.",
    "3-P": "Burst-and-trail signature showing an explosive early maximum (~200), a distinct secondary peak (~340), and a continuous elevated chemical noise baseline past index 400."
}

def generate_dual_spectrum_image_bytes(linear_spec, log_spec, title=None, dpi=60):
    """Generates a PNG byte buffer of a 2-panel vertically stacked spectrum plot (Log10 top, Linear bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=dpi)
    
    # 1. Top Panel: Log10 Spectrum
    ax1.plot(log_spec, color='black', linewidth=1.5)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlim(0, 630)
    ax1.grid(True)
    if title:
        ax1.set_title(f"{title} (Log10 Scale)")
    else:
        ax1.set_title("Log10 Scale")
        
    # 2. Bottom Panel: Linear Spectrum
    ax2.plot(linear_spec, color='black', linewidth=1.5)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlim(0, 630)
    ax2.grid(True)
    if title:
        ax2.set_title(f"{title} (Linear Scale)")
    else:
        ax2.set_title("Linear Scale")
        
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# 1. Download and Parse
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
    
    # Map labels:
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
    
    # Exclude "?" (inference class)
    df = df[df['class'] != '?'].copy()
    print(f"Remaining active spectra after mapping and filtering: {len(df)}")
    
    # Scaling and smoothing functions
    def get_log_spectrum(spectrum):
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
            
        return savgol_filter(scaled_spectrum, 5, 3).tolist()

    def get_linear_spectrum(spectrum):
        spectrum = np.array(spectrum, dtype=float)
        # Min-max scaling directly
        spec_min = np.min(spectrum)
        spec_max = np.max(spectrum)
        range_val = spec_max - spec_min
        if range_val > 0:
            scaled_spectrum = (spectrum - spec_min) / range_val
        else:
            scaled_spectrum = np.zeros_like(spectrum)
            
        return savgol_filter(scaled_spectrum, 5, 3).tolist()
        
    df['linear_spectrum'] = df['spectrum'].apply(get_linear_spectrum)
    df['log_spectrum'] = df['spectrum'].apply(get_log_spectrum)
    
    # Set spectrum to log_spectrum to maintain backward compatibility in dataframes
    df['spectrum'] = df['log_spectrum']
    
    print("Scaling and smoothing complete (Linear & Log10 columns populated).")
    return df

# 2. Split train/test
def split_train_test(df):
    train_sclks = {}
    sampled_rows = []
    
    classes = ['Noise', '1', '2', '3', '4', '5', '5-Na', '3-P']
    
    for cls in classes:
        cls_df = df[df['class'] == cls]
        # We need exactly 16 training samples for standard classes, 4 for 3-P
        n_train = 4 if cls == '3-P' else 16
        cls_train = cls_df.sample(n=n_train, random_state=123)
        train_sclks[cls] = cls_train['sclk'].tolist()
        sampled_rows.append(cls_train)
        
    train_df = pd.concat(sampled_rows, ignore_index=True)
    all_train_sclks = [sclk for sclks in train_sclks.values() for sclk in sclks]
    
    test_pool = df[~df['sclk'].isin(all_train_sclks)].copy()
    
    TEST_SIZE_PER_CLASS = 100
    test_dfs = []
    for cls in classes:
        cls_test_pool = test_pool[test_pool['class'] == cls]
        n_samples = min(TEST_SIZE_PER_CLASS, len(cls_test_pool))
        cls_test = cls_test_pool.sample(n=n_samples, random_state=42)
        test_dfs.append(cls_test)
        print(f"Included class '{cls}' in test set with {len(cls_test)} samples.")
        
    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"Train samples: {len(train_df)} (7 classes x 16 + 1 class x 4)")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, test_df, train_sclks

# 3. Generate VLM explanations concurrently with dual plot
async def generate_explanation(client, model_id, log_spectrum_array, linear_spectrum_array, label, sclk, semaphore, references=None):
    async with semaphore:
        def format_class_label(cls):
            cls_str = str(cls)
            if cls_str.lower().startswith("class"):
                return cls_str
            return f"Class {cls_str}" if cls_str.lower() != 'noise' else "Noise"

        target_label = format_class_label(label)
        img_bytes = generate_dual_spectrum_image_bytes(
            np.array(linear_spectrum_array), 
            np.array(log_spectrum_array), 
            title=f"{target_label} Sample {sclk}"
        )
        
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
    print("Pre-rendering and caching training spectrum images (Log10 + Linear)...")
    image_cache = {}
    for _, row in train_df.iterrows():
        sclk = row['sclk']
        cls = row['class']
        label = f"Class {cls}" if str(cls).lower() != 'noise' else "Noise"
        img_bytes = generate_dual_spectrum_image_bytes(
            np.array(row['linear_spectrum']), 
            np.array(row['log_spectrum']), 
            title=f"{label} Reference Sample {sclk}"
        )
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
                
        task = generate_explanation(
            client, 
            model_id, 
            row['log_spectrum'], 
            row['linear_spectrum'], 
            row['class'], 
            row['sclk'], 
            api_semaphore, 
            references=references
        )
        explanation_tasks.append(task)
            
    results = await asyncio.gather(*explanation_tasks)
    explanations_pool = [r for r in results if r is not None]
    print(f"Successfully generated {len(explanations_pool)} explanations.")
    return explanations_pool

# 4. Create consolidated batch input file
def create_ablation_batch_input_file(test_df, explanations_pool, k_values, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Generating consolidated batch request file for k={k_values}...")
    
    print("Pre-rendering and caching target spectrum images (Log10 + Linear)...")
    test_image_cache = {}
    
    for idx, row in test_df.reset_index(drop=True).iterrows():
        sclk = row['sclk']
        if sclk not in test_image_cache:
            img_bytes = generate_dual_spectrum_image_bytes(
                np.array(row['linear_spectrum']), 
                np.array(row['log_spectrum']), 
                title=f"Sample {sclk}"
            )
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            test_image_cache[sclk] = img_b64
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_df):
            print(f"  -> Pre-rendered {idx + 1}/{len(test_df)} images...")
            
    requests = []
    class_explanations = {}
    for ex in explanations_pool:
        cls = ex['label']
        if cls not in class_explanations:
            class_explanations[cls] = []
        class_explanations[cls].append(ex)
        
    for k in k_values:
        print(f"  -> Building request payloads for k={k} shots (with 4 for 3-P and 16 for other classes)...")
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
                gen_profile = GENERAL_PROFILES.get(label_key, "")
                class_label = f"Class {label_key}" if str(label_key).lower() != 'noise' else "Noise"
                few_shot_parts.append({"text": f"Example: {class_label}\\n- General Profile: {gen_profile}\\n- Specific Sample Features: {ex['explanation']}"})
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
            
    print(f"Generated JSONL file with {len(requests)} requests at {output_file}")
    return output_file

# 5. Submit and poll Vertex AI batch Prediction Job
def submit_and_poll_batch_job(configs, local_jsonl_path):
    project_id = configs.agent_settings.project_id
    bucket_name = configs.agent_settings.bucket_name
    
    filename = os.path.basename(local_jsonl_path)
    print(f"Uploading {local_jsonl_path} to GCS bucket: {bucket_name}")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
    
    gcs_input_path = f"input/{filename}"
    blob = bucket.blob(gcs_input_path)
    
    class ProgressFileWrapper(object):
        def __init__(self, fileobj, total_size):
            self.fileobj = fileobj
            self.total_size = total_size
            self.pbar = tqdm.tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc="Uploading GCS",
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
    job_display_name = f"cda-3p-ablation-dual-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
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
    
    req_file = f"batch_request_3p_dual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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

# 6. Download and Parse batch results
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
        local_temp = f"temp_3p_{os.path.basename(blob.name)}"
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
    print(f"Saved results to {output_parquet_path}")
    return merged_df

# 7. Evaluate and Save
def generate_metrics_and_plots(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    k_values = sorted(results_df['k'].unique())
    metrics_summary = {}
    
    print("\n" + "="*50 + "\n3-P focused Ablation Study (Dual Plot) Summary\n" + "="*50)
    
    for k in k_values:
        sub_df = results_df[results_df['k'] == k]
        
        # Overall accuracy
        overall_acc = accuracy_score(sub_df['true_label'], sub_df['predicted_label'])
        print(f"\nk = {k}-shot classification:")
        print(f"  Overall Accuracy: {overall_acc:.2%}")
        
        # Calculate confusion matrix for all classes
        labels = sorted(results_df['true_label'].unique())
        cm = confusion_matrix(sub_df['true_label'], sub_df['predicted_label'], labels=labels)
        
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix (k={k} shots, Log+Linear) - Acc: {overall_acc:.2%}")
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
        
    metrics_json_path = os.path.join(output_dir, "metrics_3p_ablation_dual.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nSaved metrics summary JSON to {metrics_json_path}")

async def main():
    configs = Config()
    client = genai.Client()
    model_id = configs.agent_settings.model
    
    # 1. Fetch, Parse, Scale, Smooth (both Linear and Log10 formats)
    df = download_and_preprocess_data()
    
    # 2. Split train/test
    train_df, test_df, train_sclks = split_train_test(df)
    
    # 3. Generate VLM explanations pool (with dual plots)
    explanations_pool = await generate_explanations_pool(client, model_id, train_df, train_sclks)
    
    # 4. Generate Consolidated Batch Requests JSONL File
    k_values = [4]
    output_dir = "study_results_3p_dual"
    local_jsonl = os.path.join(output_dir, "ablation_requests_3p_dual.jsonl")
    create_ablation_batch_input_file(test_df, explanations_pool, k_values, local_jsonl)
    
    # 5. Submit Batch Job to Vertex AI & Poll
    job_status = submit_and_poll_batch_job(configs, local_jsonl)
    
    # 6. Download and Parse Batch Prediction Results
    output_parquet = os.path.join(output_dir, "results_3p_ablation_dual.parquet")
    results_df = download_and_parse_batch_results(configs, job_status, test_df, output_parquet)
    
    # 7. Generate Metrics & Plots
    generate_metrics_and_plots(results_df, output_dir)
    print("3-P Ablation Study (Dual Plot) completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
