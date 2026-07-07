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

def create_batch_input_files(df, few_shot_examples, output_dir='cda_dust_agent/data/input', chunk_size=1500, limit=None, on_chunk_complete=None):
    """Creates JSONL input chunk files for Gemini Batch with Few-Shot examples, maintaining resolution dpi=60 with aggressive memory and disk management."""
    import os
    import gc
    os.makedirs(output_dir, exist_ok=True)
    
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
    print(f"Pre-rendering {total_count} spectrum plots into chunked batch JSONL files (dpi=60, chunk_size={chunk_size})...")

    chunk_results = []
    chunk_count = (total_count + chunk_size - 1) // chunk_size if total_count > 0 else 1

    fig, ax = plt.subplots(figsize=(12, 6), dpi=60)
    written_total = 0

    for chunk_idx in range(chunk_count):
        chunk_rows = target_rows[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        chunk_file = os.path.join(output_dir, f"batch_requests_part{chunk_idx + 1}.jsonl")
        print(f"Generating chunk {chunk_idx + 1}/{chunk_count} -> {chunk_file} ({len(chunk_rows)} rows)...")

        with open(chunk_file, 'w') as f_out:
            for idx, row in enumerate(chunk_rows):
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
                qi_val = row.get('qi_ampl', 'N/A')
                current_parts.append({"text": f"Sample ID (sclk): {row['sclk']} | Target Charge (qi_ampl): {qi_val} C"})

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
                written_total += 1

                if (idx + 1) % 250 == 0:
                    gc.collect()

        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=60)
        gc.collect()

        if on_chunk_complete:
            res = on_chunk_complete(chunk_file)
            chunk_results.append(res)
        else:
            chunk_results.append(chunk_file)

    plt.close('all')
    gc.collect()

    print(f"Successfully processed {written_total} requests across {len(chunk_results)} chunk operations.")
    return chunk_results, list(used_sclk_ids)


def create_batch_input_file(df, few_shot_examples, output_file='cda_dust_agent/data/input/batch_requests.jsonl', limit=None):
    output_dir = os.path.dirname(output_file) or '.'
    files, used_ids = create_batch_input_files(df, few_shot_examples, output_dir=output_dir, limit=limit)
    return files[0] if files else output_file, used_ids


def parse_response(response_text):
    """Parses JSON response from model and normalizes class labels."""
    if not response_text:
        return {"class": "Noise", "explanation": "Empty response text", "id": None}
    try:
        text = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        if "is_class_4" in data and "class_label" not in data:
            data["class_label"] = "4" if data["is_class_4"] else "Noise"
        if "class_label" in data and "class" not in data:
            data["class"] = data["class_label"]
            
        # Normalize class label formatting
        raw_label = str(data.get("class", "Noise")).strip()
        cleaned_label = raw_label.replace("Class", "").replace("class", "").strip()
        
        valid_classes = ['Noise', '1', '2', '3', '4', '5', '5-Na', '3-P']
        mapped = "Noise"
        if cleaned_label in valid_classes:
            mapped = cleaned_label
        else:
            for vc in valid_classes:
                if vc.lower() == cleaned_label.lower():
                    mapped = vc
                    break
            # Fuzzy match
            if "3-p" in cleaned_label.lower() or "3p" in cleaned_label.lower():
                mapped = "3-P"
            elif "5-na" in cleaned_label.lower() or "5na" in cleaned_label.lower():
                mapped = "5-Na"
                
        data["class"] = mapped
        return data
    except Exception as e:
        return {"class": "Noise", "explanation": f"Parse Error: {response_text}", "id": None}

def is_obvious_noise(spectrum_data, qi_ampl):
    """Evaluates if the spectrum is flat/empty noise using baseline range and peak detection."""
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks
    
    spec = np.array(spectrum_data)
    qi = float(qi_ampl) if qi_ampl is not None else 0.0
    
    # 1. Smoothed range: Noise has extremely flat baseline
    smoothed = pd.Series(spec).rolling(window=15, center=True).mean().dropna().values
    if len(smoothed) == 0:
        return True
    smoothed_range = smoothed.max() - smoothed.min()
    
    # 2. Peaks: Invert the spectrum (peaks point downward, so inverting makes them point upward)
    inverted = 1.0 - spec
    # Parameters calibrated to yield 0% False Negatives on training set
    peaks, _ = find_peaks(inverted, height=0.20, prominence=0.08, width=1)
    valid_peaks = [p for p in peaks if 20 < p < 620]
    
    # Decisions:
    # A: Smoothed baseline is extremely flat (no organic envelopes or deep hydronium valleys)
    is_flat = smoothed_range < 0.075
    # B: No peaks found and target charge is negligible
    no_peaks_low_charge = (len(valid_peaks) == 0) and (qi < 1e-15)
    
    return is_flat or no_peaks_low_charge

