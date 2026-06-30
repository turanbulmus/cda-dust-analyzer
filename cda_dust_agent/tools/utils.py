import os
import json
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt

def generate_spectrum_image_bytes(spectrum_data, title=None, dpi=60):
    """Generates a PNG byte buffer of the spectrum plot."""
    plt.figure(figsize=(12, 6), dpi=dpi)
    plt.plot(spectrum_data, color='black', linewidth=1.5)
    if title:
        plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.xlim(0, 630)
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def get_few_shot_candidates(df, n_per_class=2):
    """Extracts reference few-shot candidate rows."""
    candidates = []
    
    unique_classes = df['class'].dropna().unique()
    
    for cls in unique_classes:
        subset = df[df['class'] == cls]
        for _, row in subset.head(n_per_class).iterrows():
            cls_str = str(cls)
            label = f"Class {cls_str}" if cls_str.lower() != 'noise' else "Noise"
            candidates.append({"label": label, "sclk": row['sclk'], "spectrum": row['spectrum'].tolist()})
            
    return candidates

def create_batch_input_file(df, few_shot_examples, output_file='cda_dust_agent/data/input/batch_requests.jsonl', limit=None):
    """Creates the JSONL input file for Gemini Batch with Few-Shot examples."""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Generating batch input file: {output_file}...")
    
    used_sclk_ids = [ex['sclk'] for ex in few_shot_examples]
    
    # Needs to be imported from prompts inside the original context, handled here via relative import
    from ..prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT
    prompt_text = CLASSIFICATION_USER_PROMPT

    few_shot_parts = [{"text": prompt_text}]
    if few_shot_examples:
        few_shot_parts.append({"text": "Here are reference examples:"})
        for ex in few_shot_examples:
            few_shot_parts.append({"text": f"Example: {ex['label']} ({ex['explanation']})"})
            ex_b64 = base64.b64encode(ex['image']).decode('utf-8')
            few_shot_parts.append({"inline_data": {"mime_type": "image/png", "data": ex_b64}})
        few_shot_parts.append({"text": "Now, analyze the following spectrum:"})

    requests = []
    
    rows = df.iterrows()
    if limit:
        rows = list(rows)[:limit]
    else:
        rows = list(rows)

    for idx, row in rows:
        if row['sclk'] in used_sclk_ids:
            continue
            
        spect = row['spectrum']
        img_bytes = generate_spectrum_image_bytes(spect, title=f"Sample {row['sclk']}")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        current_parts = few_shot_parts.copy()
        current_parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        current_parts.append({"text": f"Sample ID (sclk): {row['sclk']}"})
        
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
                            "explanation": {"type": "STRING", "description": "Explanation for the prediction"}
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
            
    return output_file, used_sclk_ids

def parse_response(response_text):
    """Parses JSON response from model."""
    if not response_text:
        return {"class": "Noise", "explanation": "Empty response text", "id": None}
    try:
        text = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        if "is_class_4" in data and "class_label" not in data:
            data["class_label"] = "4" if data["is_class_4"] else "Noise"
        if "class_label" in data and "class" not in data:
            data["class"] = data["class_label"]
        return data
    except Exception as e:
        return {"class": "Noise", "explanation": f"Parse Error: {response_text}", "id": None}
