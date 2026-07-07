import os
import json
import pandas as pd

os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"

from google.cloud import storage
from cda_dust_agent.tools.utils import parse_response

def main():
    print("Connecting to GCS...")
    client = storage.Client(project="turan-genai-bb")
    bucket = client.bucket("turansgenaibb")
    blob = bucket.blob("output/prediction-model-2026-07-06T13:04:46.670249Z/predictions.jsonl")

    print("Streaming predictions directly from GCS (extracting light JSON results)...")

    records = []
    line_count = 0

    with blob.open("r", encoding="utf-8") as f:
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
                pred_class = parsed.get("class") or parsed.get("class_label") or "Noise"
                explanation = parsed.get("explanation", "")
                
                records.append({
                    "id": str(pred_id) if pred_id is not None else None,
                    "predicted_class": str(pred_class).strip(),
                    "explanation": str(explanation)
                })
            except Exception as e:
                print(f"Error parsing line {line_count}: {e}")

            if line_count % 2000 == 0:
                print(f" -> Processed {line_count} lines from GCS stream (extracted {len(records)} predictions)...")

    print(f"Done streaming! Extracted {len(records)} predictions out of {line_count} total lines.")

    # Match with ground truth
    df_gt = pd.read_parquet("cda_dust_agent/data/testing/cda_test.parquet")
    truth_map = {}
    for k, v in df_gt.set_index("sclk")["class"].to_dict().items():
        try:
            clean_k = str(int(float(k)))
        except ValueError:
            clean_k = str(k)
        truth_map[clean_k] = str(v)

    y_true = []
    y_pred = []
    explanations = []
    matched_sclks = []
    valid_classes = ['Noise', '1', '2', '3', '4', '5', '5-Na', '3-P']

    for r in records:
        pid = r["id"]
        if not pid:
            continue
        try:
            clean_id = str(int(float(pid)))
        except ValueError:
            clean_id = str(pid)
            
        if clean_id in truth_map:
            raw_label = r["predicted_class"]
            if raw_label in valid_classes:
                pred_label = raw_label
            elif "3-p" in raw_label.lower():
                pred_label = "3-P"
            elif "5-na" in raw_label.lower():
                pred_label = "5-Na"
            else:
                pred_label = "Noise"
                for cls in ['Noise', '1', '2', '3', '4', '5']:
                    if cls.lower() == raw_label.lower():
                        pred_label = cls
                        break
                        
            matched_sclks.append(clean_id)
            y_true.append(truth_map[clean_id])
            y_pred.append(pred_label)
            explanations.append(r["explanation"])

    print(f"\nSuccessfully matched {len(y_true)} predictions with ground truth!")

    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    target_names = sorted(list({str(v) for v in truth_map.values()}))
    report = classification_report(y_true, y_pred, labels=target_names, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=target_names)

    print("\n" + "="*75)
    print(" CASSINI CDA DUST AGENT: FULL DATASET (~18K) EVALUATION REPORT")
    print("="*75)
    print(report)
    print("Confusion Matrix (Rows=True, Cols=Pred):")
    print("Labels:", target_names)
    print(cm)
    print("="*75)

    os.makedirs("cda_dust_agent/data/results", exist_ok=True)
    results_df = pd.DataFrame({
        "sclk": matched_sclks,
        "true_class": y_true,
        "predicted_class": y_pred,
        "explanation": explanations
    })
    results_df.to_csv("cda_dust_agent/data/results/results.csv", index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Cassini CDA Mass Spectra Classification - Full Dataset (~18K)')
    plt.savefig("cda_dust_agent/data/results/confusion_matrix.png")
    plt.close()
    print("Saved final results to cda_dust_agent/data/results/results.csv and confusion_matrix.png")

if __name__ == "__main__":
    main()
