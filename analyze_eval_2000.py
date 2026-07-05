import os
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

dest_path = "cda_dust_agent/data/output/predictions_2000.jsonl"
print(f"Loading cached predictions from {dest_path}...")

from cda_dust_agent.tools.utils import parse_response

preds = []
with open(dest_path, 'r') as f:
    for line in f:
        if line.strip():
            preds.append(json.loads(line))

print(f"Loaded {len(preds)} prediction lines from JSONL.")

df = pd.read_parquet("cda_dust_agent/data/testing/cda_test.parquet")

def normalize_key(k):
    try:
        return str(int(float(k)))
    except (ValueError, TypeError):
        return str(k).strip()

truth_map = {normalize_key(row['sclk']): str(row['class']).strip() for _, row in df.iterrows()}

y_true = []
y_pred = []
explanations = []
matched_sclks = []

def normalize_class_label(raw):
    raw_str = str(raw).strip()
    if raw_str.startswith("Class ") or raw_str.startswith("class "):
        raw_str = raw_str[6:].strip()
    
    if raw_str.lower() == "noise":
        return "Noise"
    elif "3-p" in raw_str.lower():
        return "3-P"
    elif "5-na" in raw_str.lower():
        return "5-Na"
    elif raw_str in ["1", "2", "3", "4", "5"]:
        return raw_str
    return "Noise"

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
    if isinstance(parsed, list):
        parsed = parsed[0] if len(parsed) > 0 else {}
    elif not isinstance(parsed, dict):
        parsed = {}
        
    raw_label = parsed.get("class") or parsed.get("class_label") or "Noise"
    pred_label = normalize_class_label(raw_label)
                
    pred_id = parsed.get("id")
    explanation = parsed.get("explanation", "")
    if pred_id is not None:
        clean_id = normalize_key(pred_id)
        if clean_id in truth_map:
            matched_sclks.append(clean_id)
            y_true.append(truth_map[clean_id])
            y_pred.append(pred_label)
            explanations.append(explanation)

print(f"Matched {len(y_true)} predictions to ground truth.")

all_possible_classes = ['Noise', '1', '2', '3', '4', '5', '5-Na', '3-P']
present_labels = sorted(list(set(y_true) | set(y_pred)), key=lambda x: all_possible_classes.index(x) if x in all_possible_classes else 99)

report = classification_report(y_true, y_pred, labels=present_labels, digits=4, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=present_labels)

print("\n=================================================================")
print("     2,000 SAMPLE CLASSIFICATION REPORT (CORRECTED MAPPING)      ")
print("=================================================================")
print(report)
print("\nConfusion Matrix (Rows=True, Cols=Pred):")
print("Labels:", present_labels)
cm_df = pd.DataFrame(cm, index=present_labels, columns=present_labels)
print(cm_df)

os.makedirs("cda_dust_agent/data/results", exist_ok=True)
pd.DataFrame({
    "sclk": matched_sclks,
    "true_class": y_true,
    "predicted_class": y_pred,
    "explanation": explanations
}).to_csv("cda_dust_agent/data/results/results_2000.csv", index=False)

print("\nSaved detailed prediction results to cda_dust_agent/data/results/results_2000.csv")
