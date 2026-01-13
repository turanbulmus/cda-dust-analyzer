
import os
import pandas as pd
import random
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
from datetime import datetime

# Reuse image generation from batch_classify to ensure consistency
from batch_classify import generate_spectrum_image_bytes

# Load .env
load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = "us-central1"
MODEL_ID = "gemini-3-pro-preview"

def analyze_samples():
    print("Starting Prompt Optimization Pipeline...")
    
    # Init Client
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
    if not api_key:
        print("Error: GOOGLE_CLOUD_API_KEY not found.")
        return
    client = genai.Client(vertexai=True, api_key=api_key)

    # Load Data
    try:
        df = pd.read_parquet('data/cda_sample.parquet')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Sample Data
    classes = ['4', '1', 'Noise']
    samples_per_class = 4
    
    analysis_results = {}

    for cls in classes:
        print(f"\nAnalyzing Class {cls} ({samples_per_class} samples)...")
        # Handle string/int mixing in column
        if cls == 'Noise':
            subset = df[df['class'] == 'Noise']
        else:
            subset = df[df['class'].astype(str) == cls]
            
        if subset.empty:
            print(f"Warning: No samples found for Class {cls}")
            continue
            
        # Random sample
        selected = subset.sample(min(len(subset), samples_per_class), random_state=42)
        
        descriptions = []
        
        for idx, row in selected.iterrows():
            s_id = row['sclk']
            img_bytes = generate_spectrum_image_bytes(row['spectrum'], title=f"Class {cls} Sample {s_id}")
            
            # Analyze this specific image
            prompt = f"""Analyze this spectrum plot for a Class '{cls}' event.
            NOTE: The X-axis represents Time (Time-of-Flight) and the Y-axis represents Amplitude (Signal Intensity).
            
            Describe the key visual features, specifically:
            1. Presence and location (Time/X) of sharp peaks (High Amplitude/Y).
            2. General noise level (Baseline Amplitude).
            3. Any activity in the 200-400 index range.
            4. Any activity in the 0-50 index range.
            5. Any activity around index 640.
            
            Focus on the SHAPE and INTENSITY of signals. Do not overly fixate on exact X-indices if the feature shape is robust but slightly time-shifted.
            Keep it concise."""
            
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=prompt),
                                types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes))
                            ]
                        )
                    ],
                    config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024)
                )
                if response.text:
                    descriptions.append(f"Sample {s_id}: {response.text}")
                time.sleep(1) # Rate limit courtesy
            except Exception as e:
                print(f"Error analyzing sample {s_id}: {e}")
        
        analysis_results[cls] = "\n".join(descriptions)
        print(f"Completed analysis for Class {cls}")

    # Phase 2: Synthesize Prompt
    print("\nSynthesizing Optimized Prompt...")
    
    synthesis_prompt = """You are a prompt engineer for a cosmic dust classification system.
    I have analyzed 10 samples each from Class 4 (Target), Class 1 (Distractor), and Noise (Distractor).
    
    Here are the observations:
    
    ### Class 4 Observations
    {c4_obs}
    
    ### Class 1 Observations
    {c1_obs}
    
    ### Noise Observations
    {noise_obs}
    
    *** CRITICAL FEEDBACK FROM PREVIOUS MODEL RUN ***
    The previous prompt failed significantly by classifying Class 4 and Class 1 samples as "Noise". 
    It suffered from "Conservative Bias", where any spectral complexity was dismissed as "chaotic static".
    
    Specific Failure Examples:
    1. Class 4 misclassified as Noise: The model saw a "single high-amplitude spike" and "chaotic low-level static" but missed the mid-range peaks because they weren't "distinct" enough for its strict criteria.
    2. Class 1 misclassified as Noise: The model rejected Class 1 because the "Early Spike" wasn't seen as the *identifying* feature, but rather as just an artifact, and the rest was "featureless".
    
    CORRECTION REQUIRED:
    - You MUST explicitly define "Noise" as having *truly* no structure (flat baseline). If there is "hairy" or "messy" signal in the mid-range (indices 150-500), it is likely Class 4, NOT Noise.
    - Class 4 peaks might be embedded in some noise. Do not require "perfect clean peaks". 
    - Differentiate Class 1 from Noise: Class 1 has a *very strong* start spike (often >2x background) and a *relatively* quiet mid-range, but maybe not perfectly flat.
    
    Based on these observations and corrections, identify the MOST ROBUST distinguishing features for Class 4.
    
    CRITICAL: 
    - X-axis is Time (Time-of-Flight). Features might jitter/shift slightly in time.
    - Y-axis is Amplitude. Look for high signal-to-noise ratio features.
    - Don't just rely on rigid X-indices; describe the SHAPE and RELATIVE locations if relevant.
    
    Write a SYSTEM INSTRUCTION and a USER PROMPT that:
    1. Clearly defines Class 4 criteria based on the verified features (e.g., peak locations).
    2. Explicitly lists rejection criteria for features common in Class 1 or Noise but absent in Class 4.
    3. Is concise and unambiguous.
    
    Output format:
    ---SYSTEM_INSTRUCTION_START---
    [Your System Instruction Here]
    ---SYSTEM_INSTRUCTION_END---
    
    ---USER_PROMPT_START---
    [Your User Prompt Here]
    ---USER_PROMPT_END---
    """.format(
        c4_obs=analysis_results.get('4', 'No data'),
        c1_obs=analysis_results.get('1', 'No data'),
        noise_obs=analysis_results.get('Noise', 'No data')
    )
    
    try:
        final_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=synthesis_prompt)])],
            config=types.GenerateContentConfig(temperature=0.5, max_output_tokens=8192)
        )
        
        if final_response.text:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Archive prompt and examples
            archive_dir = "archive"
            os.makedirs(archive_dir, exist_ok=True)
            
            # Save raw text
            filename = f"{archive_dir}/optimized_prompt_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(final_response.text)
                
            # Save structured archive
            archive_data = {
                "timestamp": timestamp,
                "model_id": MODEL_ID,
                "generated_prompt": final_response.text,
                "observations": analysis_results
            }
            archive_json = f"{archive_dir}/optimize_run_{timestamp}.json"
            with open(archive_json, "w") as f:
                json.dump(archive_data, f, indent=2)
                
            print(f"\nOptimized prompt saved to: {filename}")
            print(f"Archive data saved to: {archive_json}")
            print("-" * 40)
            print(final_response.text)
            print("-" * 40)
            
            # --- AUTO-UPDATE PIPELINE ---
            try:
                print("\nInitiating Auto-Update of batch_classify.py...")
                
                # 1. Parse Response
                import re
                
                sys_instruction_match = re.search(r'---SYSTEM_INSTRUCTION_START---(.*?)---SYSTEM_INSTRUCTION_END---', final_response.text, re.DOTALL)
                user_prompt_match = re.search(r'---USER_PROMPT_START---(.*?)---USER_PROMPT_END---', final_response.text, re.DOTALL)
                
                if sys_instruction_match and user_prompt_match:
                    new_sys_inst = sys_instruction_match.group(1).strip()
                    new_user_prompt = user_prompt_match.group(1).strip()
                    
                    target_file = 'batch_classify.py'
                    with open(target_file, 'r') as f:
                        code = f.read()
                    
                    # Backup
                    backup_file = f"{target_file}.bak"
                    with open(backup_file, 'w') as f:
                        f.write(code)
                    print(f"Backup created at {backup_file}")
                    
                    # Replace SYSTEM_INSTRUCTION_TEXT
                    # Look for SYSTEM_INSTRUCTION_TEXT = """..."""
                    # We use a robust regex that handles potential multiline strings
                    
                    # Note: expecting triple quotes in the target file
                    code = re.sub(
                        r'SYSTEM_INSTRUCTION_TEXT = """(.*?)"""', 
                        f'SYSTEM_INSTRUCTION_TEXT = """{new_sys_inst}"""', 
                        code, 
                        flags=re.DOTALL
                    )
                    
                    # Replace USER_PROMPT_TEXT
                    code = re.sub(
                        r'USER_PROMPT_TEXT = """(.*?)"""', 
                        f'USER_PROMPT_TEXT = """{new_user_prompt}"""', 
                        code, 
                        flags=re.DOTALL
                    )
                    
                    with open(target_file, 'w') as f:
                        f.write(code)
                        
                    print(f"SUCCESS: {target_file} updated with new optimal prompt.")
                    
                else:
                    print("Error: Could not parse prompt blocks from synthesis response. Update skipped.")
                    print("Ensure the model output contains ---SYSTEM_INSTRUCTION_START--- blocks.")
                    
            except Exception as update_e:
                print(f"Error during auto-update: {update_e}")
                
        else:
            print("Error: Empty response for synthesis.")
            print(f"Response Candidates: {final_response.candidates}")
            if final_response.prompt_feedback:
                print(f"Prompt Feedback: {final_response.prompt_feedback}")
            
    except Exception as e:
        print(f"Error during synthesis: {e}")

if __name__ == "__main__":
    analyze_samples()
