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

def get_few_shot_candidates(df, n_per_class=16):
    """Extracts reference few-shot candidate rows, matching notebook 10 sampling strategy (16 per standard class, 4 for 3-P)."""
    candidates = []
    
    unique_classes = df['class'].dropna().unique()
    
    for cls in unique_classes:
        cls_str = str(cls)
        n_samples = 4 if cls_str == "3-P" else (n_per_class if isinstance(n_per_class, int) else 16)
        subset = df[df['class'] == cls]
        for _, row in subset.head(n_samples).iterrows():
            label = f"Class {cls_str}" if cls_str.lower() != 'noise' else "Noise"
            candidates.append({"label": label, "raw_class": cls_str, "sclk": row['sclk'], "spectrum": row['spectrum'].tolist()})
            
    return candidates

def create_batch_input_file(df, few_shot_examples, output_file='cda_dust_agent/data/input/batch_requests.jsonl', limit=None):
    """Creates the JSONL input file for Gemini Batch with Few-Shot examples, using fast canvas rendering."""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Generating batch input file: {output_file}...")
    
    used_sclk_ids = set(ex['sclk'] for ex in few_shot_examples)
    
    from ..prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT, GENERAL_PROFILES
    prompt_text = CLASSIFICATION_USER_PROMPT

    few_shot_parts = [{"text": prompt_text}]
    if few_shot_examples:
        few_shot_parts.append({"text": "Here are reference examples:"})
        for ex in few_shot_examples:
            raw_key = ex.get('raw_class', ex['label'].replace("Class ", "").strip())
            gen_profile = GENERAL_PROFILES.get(raw_key, "")
            few_shot_parts.append({"text": f"Example: {ex['label']}\n- General Profile: {gen_profile}\n- Specific Sample Features: {ex['explanation']}"})
            ex_b64 = base64.b64encode(ex['image']).decode('utf-8') if isinstance(ex['image'], bytes) else ex['image_base64']
            few_shot_parts.append({"inline_data": {"mime_type": "image/png", "data": ex_b64}})
        few_shot_parts.append({"text": "Now, analyze the following spectrum:"})

    rows = df.iterrows()
    if limit:
        target_rows = [row for _, row in rows if row['sclk'] not in used_sclk_ids][:limit]
    else:
        target_rows = [row for _, row in rows if row['sclk'] not in used_sclk_ids]

    total_count = len(target_rows)
    print(f"Pre-rendering {total_count} spectrum plots into batch JSONL...")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=60)
    written_count = 0

    with open(output_file, 'w') as f_out:
        for idx, row in enumerate(target_rows):
            ax.clear()
            ax.plot(row['spectrum'], color='black', linewidth=1.5)
            ax.set_title(f"Sample {row['sclk']}")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0, 630)
            ax.grid(True)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

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
            f_out.write(json.dumps(request) + '\n')
            written_count += 1

            if (idx + 1) % 200 == 0 or (idx + 1) == total_count:
                print(f"  -> Generated & written {idx + 1}/{total_count} requests...")

    plt.close(fig)
    print(f"Successfully generated {written_count} requests in {output_file}")
    return output_file, list(used_sclk_ids)


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
