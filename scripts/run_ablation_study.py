import os
import json
import base64
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from google import genai
from google.genai import types
import huggingface_hub

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cda_dust_agent.config import Config
from cda_dust_agent.prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT, ANNOTATION_USER_PROMPT
from cda_dust_agent.tools.utils import generate_spectrum_image_bytes

async def generate_explanation(client, model_id, spectrum_array, label, sclk, semaphore):
    async with semaphore:
        img_bytes = generate_spectrum_image_bytes(np.array(spectrum_array), title=f"{label} Sample {sclk}")
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

def preprocess_huggingface_data():
    print("Fetching full dataset from HuggingFace...")
    REPO_ID = "CosmicDustGroup/cassini-cda-spectra"
    FILENAME_TRAIN = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"
    
    file_train_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_TRAIN, repo_type="dataset")
    
    full_df = pd.read_parquet(file_train_path)
    
    if 'qi_ampl' in full_df.columns:
        full_df = full_df[full_df['qi_ampl'] >= 10 * 10**-15]
        
    full_df = full_df[full_df['spectrum'].apply(len) == 1018].copy()
    
    def crop_spectrum(spectrum):
        return spectrum[10:641]
    full_df['spectrum'] = full_df['spectrum'].apply(crop_spectrum)
    
    full_df['class'] = full_df['class'].apply(lambda x: '?' if isinstance(x, str) and "X" in x else x)
    full_df = full_df[full_df['class'] != '?'].copy()
    
    def qm_scaling(spectrum):
        spectrum = np.log10(spectrum + np.abs(np.min(spectrum)))
        spectrum = np.nan_to_num(spectrum, neginf=0)
        max_val = np.max(spectrum)
        min_val = np.min(spectrum)
        if max_val > min_val:
            spectrum = (spectrum - min_val) / (max_val - min_val)
        return spectrum

    full_df['spectrum'] = full_df['spectrum'].apply(qm_scaling)
    
    return full_df

async def run_ablation_study():
    k_values = [1, 2, 4, 8, 16, 32, 64]
    
    config = Config()
    client = genai.Client()
    model_id = config.agent_settings.model
    
    full_df = preprocess_huggingface_data()
    print(f"Total labeled, filtered rows available for ablation: {len(full_df)}")
    
    # We will use a semaphore to avoid hitting Gemini API rate limits immediately
    api_semaphore = asyncio.Semaphore(5)
    
    for k in k_values:
        print(f"\n{'='*40}\nStarting ablation study for k={k}\n{'='*40}")
        output_dir = f"cda_dust_agent/data/ablation_study/k_{k}"
        os.makedirs(output_dir, exist_ok=True)
        results_csv_path = os.path.join(output_dir, f"results_k{k}.csv")
        
        class_counts = full_df['class'].value_counts()
        
        # We need at least k for training PLUS some minimum for testing
        min_test_samples = 1 
        viable_classes = []
        for cls, count in class_counts.items():
            if count >= (k + min_test_samples):
                viable_classes.append(cls)
                
        if not viable_classes:
            print(f"No viable classes found for k={k}. Skipping.")
            continue
            
        print(f"Viable classes for k={k}: {viable_classes}")
        
        k_shot_examples = []
        used_sclks = []
        
        explanation_tasks = []
        
        # Select exactly k items per viable class for "few-shot training"
        for cls in viable_classes:
            class_pool = full_df[full_df['class'] == cls].sample(n=k, random_state=42)
            
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
        
        # The remaining items (ALL of them, excluding those used for few-shot) become the testing set!
        batch_test_df = full_df[full_df['class'].isin(viable_classes)].copy()
        batch_test_df = batch_test_df[~batch_test_df['sclk'].isin(used_sclks)]
        
        print(f"Executing inference on {len(batch_test_df)} test spectra for k={k}...")
        
        inference_tasks = []
        for _, row in batch_test_df.iterrows():
            task = classify_spectrum(client, model_id, row, k_shot_examples, api_semaphore)
            inference_tasks.append(task)
            
        # Run inferences concurrently (semaphore limits to 5 at a time)
        all_results = []
        chunk_size = 500
        for i in range(0, len(inference_tasks), chunk_size):
            chunk = inference_tasks[i:i + chunk_size]
            print(f"Processing inference chunk {i} to {i+len(chunk)-1} / {len(inference_tasks)}...")
            chunk_results = await asyncio.gather(*chunk)
            all_results.extend(chunk_results)
            
            # Save progressively so data isn't lost if it crashes
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(results_csv_path, index=False)
                
        print(f"Saved inference results for k={k} to {results_csv_path}")

if __name__ == "__main__":
    asyncio.run(run_ablation_study())