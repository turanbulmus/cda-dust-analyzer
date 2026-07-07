import os
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from cda_dust_agent.tools.utils import parse_response

def normalize_class_label(raw_label):
    if not raw_label:
        return "Noise"
    raw_label = str(raw_label).strip().replace("Class", "").replace("class", "").strip()
    
    valid_classes = ['Noise', '1', '2', '3', '4', '5', '5-Na', '3-P']
    
    # Exact check first
    if raw_label in valid_classes:
        return raw_label
        
    # Case-insensitive checks
    for vc in valid_classes:
        if vc.lower() == raw_label.lower():
            return vc
            
    # Fuzzy checks for custom formats
    if "3-p" in raw_label.lower() or "3p" in raw_label.lower():
        return "3-P"
    if "5-na" in raw_label.lower() or "5na" in raw_label.lower():
        return "5-Na"
        
    return "Noise"

def main():
    results_dir = "cda_dust_agent/data/results"
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "results.csv")
    predictions_jsonl = "cda_dust_agent/data/output/predictions.jsonl"
    
    print("Loading ground truth cda_test.parquet...")
    df_gt = pd.read_parquet("cda_dust_agent/data/testing/cda_test.parquet")
    truth_map = {}
    for k, v in df_gt.set_index("sclk")["class"].to_dict().items():
        try:
            clean_k = str(int(float(k)))
        except ValueError:
            clean_k = str(k)
        truth_map[clean_k] = str(v)
    print(f"Ground truth loaded with {len(truth_map)} spectra.")

    print(f"Parsing local predictions file: {predictions_jsonl}...")

    records = []
    line_count = 0

    with open(predictions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                resp_text = ""
                if "response" in data:
                    candidates = data["response"].get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            resp_text = parts[0].get("text", "")
                elif "predictions" in data:
                    candidates = data["predictions"][0].get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            resp_text = parts[0].get("text", "")
                
                parsed = parse_response(resp_text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    parsed = parsed[0]
                elif not isinstance(parsed, dict):
                    parsed = {}
                    
                pred_id = parsed.get("id")
                if pred_id is None:
                    continue
                    
                try:
                    clean_id = str(int(float(pred_id)))
                except ValueError:
                    clean_id = str(pred_id)
                
                if clean_id in truth_map:
                    raw_label = str(parsed.get("class") or parsed.get("class_label") or "Noise").strip()
                    pred_label = normalize_class_label(raw_label)
                    
                    records.append({
                        "sclk": clean_id,
                        "true_class": truth_map[clean_id],
                        "predicted_class": pred_label,
                        "explanation": str(parsed.get("explanation", ""))
                    })

            except Exception as e:
                print(f"Error parsing line {line_count}: {e}")

            if line_count % 3000 == 0:
                print(f" -> Line {line_count} parsed (Extracted={len(records)})...")

    print(f"Finished parsing local predictions! Total parsed: {len(records)} out of {line_count} lines.")

    # Write all results to results.csv
    results_df = pd.DataFrame(records)
    results_df.to_csv(results_csv, index=False)
    print(f"Saved complete results to {results_csv}")

    # Load results for metrics
    y_true = results_df["true_class"].astype(str).tolist()
    y_pred = results_df["predicted_class"].astype(str).tolist()

    target_names = sorted(list({str(v) for v in truth_map.values()}))
    report = classification_report(y_true, y_pred, labels=target_names, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=target_names)

    print("\n" + "="*75)
    print(" CASSINI CDA DUST AGENT: FULL DATASET (~18K) EVALUATION REPORT (FIXED PARSING)")
    print("="*75)
    print(report)
    print("Confusion Matrix (Rows=True, Cols=Pred):")
    print("Labels:", target_names)
    print(cm)
    print("="*75)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Cassini CDA Mass Spectra Classification - Full Dataset (~18K)')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    print("Saved final results plot to confusion_matrix.png")

if __name__ == "__main__":
    main()
