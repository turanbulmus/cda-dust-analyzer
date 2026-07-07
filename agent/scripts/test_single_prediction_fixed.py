import os
import json
import base64
import numpy as np
import pandas as pd
from google import genai
from google.genai import types

from cda_dust_agent.config import Config
from cda_dust_agent.tools.utils import generate_spectrum_image_bytes
from cda_dust_agent.prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT, GENERAL_PROFILES

os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"

def main():
    configs = Config()
    cache_file = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
    
    print("Loading regenerated clean few-shot examples...")
    few_shot_examples = []
    with open(cache_file, "r") as f:
        for line in f:
            if line.strip():
                few_shot_examples.append(json.loads(line))
    print(f"Loaded {len(few_shot_examples)} clean few-shot examples.")

    # Select one target spectrum (e.g. Class 4 Mineral spectrum, sclk 1543798770)
    print("Selecting target spectrum for single test prediction...")
    df_train = pd.read_parquet("cda_dust_agent/data/testing/cda_test.parquet")
    target_row = df_train[df_train["sclk"] == 1543798770].iloc[0]
    
    target_sclk = int(target_row["sclk"])
    target_spectrum = np.array(target_row["spectrum"])
    target_label = target_row["class"]
    
    print(f"Target selected: sclk={target_sclk}, true class={target_label}")
    
    # Generate target spectrum plot PNG
    target_img_bytes = generate_spectrum_image_bytes(target_spectrum, title=f"Sample {target_sclk}")
    target_img_b64 = base64.b64encode(target_img_bytes).decode("utf-8")

    # Build prompt parts list matching Gemini Batch submission format
    prompt_text = CLASSIFICATION_USER_PROMPT
    contents_parts = [{"text": prompt_text}]
    
    contents_parts.append({"text": "Here are reference examples:"})
    for ex in few_shot_examples:
        raw_key = ex.get('raw_class', ex['label'].replace("Class ", "").strip())
        gen_profile = GENERAL_PROFILES.get(raw_key, "")
        contents_parts.append({
            "text": f"Example: {ex['label']}\n- General Profile: {gen_profile}\n- Specific Sample Features: {ex['explanation']}"
        })
        contents_parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": ex["image_base64"]
            }
        })
    
    contents_parts.append({"text": "Now, analyze the following spectrum:"})
    contents_parts.append({
        "inline_data": {
            "mime_type": "image/png",
            "data": target_img_b64
        }
    })
    
    qi_val = target_row.get("qi_ampl", "N/A")
    contents_parts.append({"text": f"Sample ID (sclk): {target_sclk} | Target Charge (qi_ampl): {qi_val} C"})

    # Initialize Gemini client
    project = "turan-genai-bb"
    model_id = "gemini-2.5-flash"
    print(f"Initializing genai.Client (project={project}, model={model_id})...")
    client = genai.Client(vertexai=True, project=project, location="us-central1")
    
    # Convert parts list to SDK Part objects
    sdk_parts = []
    for p in contents_parts:
        if "text" in p:
            sdk_parts.append(types.Part.from_text(text=p["text"]))
        elif "inline_data" in p:
            sdk_parts.append(types.Part.from_bytes(
                data=base64.b64decode(p["inline_data"]["data"]),
                mime_type=p["inline_data"]["mime_type"]
            ))

    print("Sending content generation request to Vertex AI...")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[types.Content(role="user", parts=sdk_parts)],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION_TEXT
            )
        )
        print("Success! Received response.")
        
        # Structure it exactly like Vertex AI Batch Predictions line output
        output_data = {
            "request": {
                "contents": [
                    {
                        "parts": contents_parts
                    }
                ],
                "systemInstruction": {
                    "parts": [
                        {
                            "text": SYSTEM_INSTRUCTION_TEXT
                        }
                    ]
                }
            },
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": response.text
                                }
                            ],
                            "role": "model"
                        },
                        "finishReason": "STOP"
                    }
                ]
            }
        }
        
        local_dest = "cda_dust_agent/data/output/sample_prediction.json"
        os.makedirs(os.path.dirname(local_dest), exist_ok=True)
        with open(local_dest, "w") as out:
            json.dump(output_data, out, indent=2)
        print(f"Saved clean sample prediction payload to {local_dest}!")
        
    except Exception as e:
        print("Failed to run prediction call:", e)

if __name__ == "__main__":
    main()
