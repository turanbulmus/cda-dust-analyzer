
import os
import pandas as pd
import random
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
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
    samples_per_class = 10
    
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
    
    Based on these observations, identify the MOST ROBUST distinguishing features for Class 4.
    
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
            config=types.GenerateContentConfig(temperature=0.5, max_output_tokens=2048)
        )
        
        if final_response.text:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"optimized_prompt_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(final_response.text)
            print(f"\nOptimized prompt saved to: {filename}")
            print("-" * 40)
            print(final_response.text)
            print("-" * 40)
        else:
            print("Error: Empty response for synthesis.")
            
    except Exception as e:
        print(f"Error during synthesis: {e}")

if __name__ == "__main__":
    analyze_samples()
