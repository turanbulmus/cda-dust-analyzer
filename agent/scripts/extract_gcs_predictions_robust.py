import os
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"

from google.cloud import storage
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

    # Load already processed SCLK IDs if CSV exists (to enable resume)
    processed_sclks = {}
    if os.path.exists(results_csv) and os.path.getsize(results_csv) > 0:
        try:
            existing_df = pd.read_csv(results_csv)
            if not existing_df.empty and "sclk" in existing_df.columns:
                for idx, row in existing_df.iterrows():
                    processed_sclks[str(int(float(row["sclk"])))] = {
                        "true_class": str(row["true_class"]),
                        "predicted_class": str(row["predicted_class"]),
                        "explanation": str(row["explanation"])
                    }
                print(f"Resuming: found {len(processed_sclks)} already processed predictions in {results_csv}")
        except Exception as e:
            print(f"Could not load existing results.csv, starting fresh. Error: {e}")

    print("Connecting to GCS...")
    client = storage.Client(project="turan-genai-bb")
    bucket = client.bucket("turansgenaibb")
    blob = bucket.blob("output/prediction-model-2026-07-06T13:04:46.670249Z/predictions.jsonl")
    
    # 128 MB chunk size for high-throughput buffering
    buffer_chunk_size = 128 * 1024 * 1024 
    print(f"Streaming predictions directly from GCS with chunk_size={buffer_chunk_size//(1024*1024)}MB...")

    new_records = []
    line_count = 0
    skipped_count = 0

    with blob.open("r", encoding="utf-8", chunk_size=buffer_chunk_size) as f:
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
                
                # Skip if already processed in previous crashed runs
                if clean_id in processed_sclks:
                    skipped_count += 1
                    continue
                
                if clean_id in truth_map:
                    raw_label = str(parsed.get("class") or parsed.get("class_label") or "Noise").strip()
                    pred_label = normalize_class_label(raw_label)
                                
                    rec = {
                        "sclk": clean_id,
                        "true_class": truth_map[clean_id],
                        "predicted_class": pred_label,
                        "explanation": str(parsed.get("explanation", ""))
                    }
                    new_records.append(rec)
                    processed_sclks[clean_id] = rec

            except Exception as e:
                print(f"Error parsing line {line_count}: {e}")

            if line_count % 1000 == 0:
                print(f" -> Line {line_count} parsed (Extracted={len(processed_sclks)}, New={len(new_records)}, Skipped={skipped_count})...")
                # Flush newly processed chunks incrementally to disk
                if new_records:
                    temp_df = pd.DataFrame(new_records)
                    header_needed = not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0
                    temp_df.to_csv(results_csv, mode='a', index=False, header=header_needed)
                    new_records.clear()

    # Flush remaining records
    if new_records:
        temp_df = pd.DataFrame(new_records)
        header_needed = not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0
        temp_df.to_csv(results_csv, mode='a', index=False, header=header_needed)
        new_records.clear()

    print(f"Streaming finished! Total lines processed: {line_count}. Total parsed predictions: {len(processed_sclks)}")

    # Load combined results for metrics calculation
    final_df = pd.read_csv(results_csv)
    y_true = final_df["true_class"].astype(str).tolist()
    y_pred = final_df["predicted_class"].astype(str).tolist()

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

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Cassini CDA Mass Spectra Classification - Full Dataset (~18K)')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Saved evaluation metrics to {results_csv} and confusion_matrix.png")

if __name__ == "__main__":
    main()
