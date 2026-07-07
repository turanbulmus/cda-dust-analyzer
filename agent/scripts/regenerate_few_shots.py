import os
import json
import base64
import re
import shutil
import numpy as np
import pandas as pd
from google import genai
from google.genai import types

from cda_dust_agent.config import Config
from cda_dust_agent.tools.utils import get_few_shot_candidates, generate_spectrum_image_bytes
from cda_dust_agent.prompts import ANNOTATION_USER_PROMPT, SYSTEM_INSTRUCTION_TEXT

os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"

def main():
    configs = Config()
    cache_dir = "cda_dust_agent/data/input/examples"
    cache_file = os.path.join(cache_dir, "cached_examples.jsonl")
    
    print("Cleaning up old error-ridden few-shot files...")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Removed {cache_file}")
        
    store_dir = "cda_dust_agent/data/input/store"
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
        print(f"Cleared directory {store_dir}")
    os.makedirs(store_dir, exist_ok=True)
    
    annotated_dir = "cda_dust_agent/data/annotated_spectra"
    if os.path.exists(annotated_dir):
        shutil.rmtree(annotated_dir)
        print(f"Cleared directory {annotated_dir}")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading training dataset Parquet...")
    df = pd.read_parquet("cda_dust_agent/data/raw/cda_train.parquet")
    
    n_per_class = configs.agent_settings.few_shot_n
    print(f"Selecting few-shot candidates (n_per_class={n_per_class})...")
    candidates = get_few_shot_candidates(df, n_per_class=n_per_class)
    print(f"Total candidates selected: {len(candidates)}")
    
    project = "turan-genai-bb"
    model_id = "gemini-2.5-flash"
    
    print(f"Initializing Vertex AI Client (project={project}, model={model_id})...")
    client = genai.Client(vertexai=True, project=project, location="us-central1")
    
    few_shot_examples = []
    
    for i, cand in enumerate(candidates):
        print(f"Annotating candidate {i+1}/{len(candidates)}: {cand['label']} (sclk: {cand['sclk']})...")
        
        # Generate plot image
        img_bytes = generate_spectrum_image_bytes(np.array(cand["spectrum"]), title=f"{cand['label']} Sample {cand['sclk']}")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt = ANNOTATION_USER_PROMPT.format(label=cand['label'])
        contents = [
            types.Content(role="user", parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            ])
        ]
        
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION_TEXT
                )
            )
            explanation = response.text.strip()
            print(" -> Success! Explanation:", explanation[:120], "...")
        except Exception as e:
            explanation = f"Failed to generate explanation: {e}"
            print(" -> FAILED! Error:", e)
            
        new_example = {
            "label": cand["label"],
            "raw_class": cand["raw_class"],
            "image_base64": img_b64,
            "explanation": explanation,
            "sclk": int(cand["sclk"])
        }
        
        # Write to cached_examples.jsonl (to restore cache state)
        with open(cache_file, "a") as f:
            f.write(json.dumps(new_example) + "\n")
            
        # Write PNG to annotated_spectra
        safe_class = re.sub(r'[^\w\s-]', '', cand['label']).replace(' ', '_')
        with open(os.path.join(annotated_dir, f"{cand['sclk']}_{safe_class}.png"), "wb") as f:
            f.write(img_bytes)
            
        # Write PNG to store
        with open(os.path.join(store_dir, f"{cand['sclk']}_{safe_class}.png"), "wb") as f:
            f.write(img_bytes)
            
        # Write metadata JSON to store
        with open(os.path.join(store_dir, f"{cand['sclk']}_{safe_class}.json"), "w") as f:
            json.dump({
                "sclk": int(cand["sclk"]),
                "label": cand["label"],
                "raw_class": cand["raw_class"],
                "prompt": prompt,
                "explanation": explanation
            }, f, indent=2)

    print("\n" + "="*50)
    print(" FEW-SHOT EXAMPLES REGENERATION COMPLETE!")
    print(f" Saved to {cache_file}")
    print("="*50)

if __name__ == "__main__":
    main()
