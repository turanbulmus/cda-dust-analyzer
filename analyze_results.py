import pandas as pd
import json
import hashlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
import os

from batch_classify import generate_spectrum_image_bytes
import base64

def get_last_image_data(request_dict):
    """Extracts the last inline_data ('data') from the request to use as a key."""
    last_data = None
    try:
        # Check standard recursive structure
        if 'contents' in request_dict:
             for content in request_dict['contents']:
                if 'parts' in content:
                    for part in content['parts']:
                        if 'inline_data' in part and part['inline_data'] and 'data' in part['inline_data']:
                            last_data = part['inline_data']['data']
        # Check flat structure
        if 'inline_data' in request_dict:
             last_data = request_dict['inline_data']['data']
             
        return last_data
    except Exception:
        return None

def parse_model_response(response_text):
    """Parses the model's JSON response."""
    if not response_text:
        return False, "Empty Response"
        
    cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned_text)
        is_class_4 = data.get("is_class_4", False)
        reasoning = data.get("reasoning", "")
        return is_class_4, reasoning
    except Exception as e:
        return False, f"JSON Parse Error: {e} | Text: {response_text[:100]}..."

def main():
    print("Loading Ground Truth...")
    try:
        df = pd.read_parquet('data/cda_sample.parquet')
    except Exception as e:
        print(f"Error loading data/cda_sample.parquet: {e}")
        return

    # Map Image Data (Base64) -> Index in DF
    request_map = {}
    
    # Strategy 1: Load/Download batch_requests.jsonl
    if not os.path.exists('batch_requests.jsonl'):
        print("batch_requests.jsonl not found locally. Attempting to download from GCS...")
        try:
            from google.cloud import storage
            from dotenv import load_dotenv
            load_dotenv()
            
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            bucket_name = os.environ.get("GCS_BUCKET_NAME")
            
            if project_id and bucket_name:
                storage_client = storage.Client(project=project_id)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob("input/batch_requests.jsonl")
                blob.download_to_filename("batch_requests.jsonl")
                print("Successfully downloaded batch_requests.jsonl from GCS.")
            else:
                print("Warning: Missing GOOGLE_CLOUD_PROJECT or GCS_BUCKET_NAME in .env. Cannot download.")
        except Exception as e:
            print(f"Failed to download batch_requests.jsonl: {e}")

    if os.path.exists('batch_requests.jsonl'):
        print("Building Request Map from batch_requests.jsonl...")
        with open('batch_requests.jsonl', 'r') as f:
            for idx, line in enumerate(f):
                if idx >= len(df):
                    break
                try:
                    req_obj = json.loads(line)
                    if 'request' in req_obj:
                        img_data = get_last_image_data(req_obj['request'])
                        if img_data:
                            request_map[img_data] = idx
                except Exception as e:
                    pass
        print(f"Mapped {len(request_map)} unique requests from file.")

    # Strategy 2: If File Missing or Map Empty, Re-generate from DF (Robust)
    if not request_map:
        print("batch_requests.jsonl not found or empty. Re-generating request map from dataframe...")
        # We need to match the EXACT image generation used in batch_classify.py
        for idx, row in df.iterrows():
            try:
                # Same title format as batch_classify.py: f"Sample {row['sclk']}"
                img_bytes = generate_spectrum_image_bytes(row['spectrum'], title=f"Sample {row['sclk']}")
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                request_map[img_b64] = idx
            except Exception as e:
                print(f"Error generating map for index {idx}: {e}")
                
        print(f"Re-generated map for {len(request_map)} items.")

    print("Processing Predictions from predictions.jsonl...")
    if not os.path.exists('predictions.jsonl'):
        print("predictions.jsonl not found.")
        return

    results = []
    
    with open('predictions.jsonl', 'r') as f:
        for line_no, line in enumerate(f):
            try:
                pred_obj = json.loads(line)
                
                if 'error' in pred_obj:
                    # print(f"Prediction Error on line {line_no}: {pred_obj['error']}")
                    continue
                
                if 'request' not in pred_obj:
                    continue
                
                img_data = get_last_image_data(pred_obj['request'])
                
                if not img_data or img_data not in request_map:
                    # print(f"Warning: Could not match prediction {line_no} to input request.")
                    continue
                    
                idx = request_map[img_data]
                row = df.iloc[idx]
                
                response_text = ""
                if 'response' in pred_obj and 'candidates' in pred_obj['response']:
                    cands = pred_obj['response']['candidates']
                    if cands and 'content' in cands[0] and 'parts' in cands[0]['content']:
                        response_text = cands[0]['content']['parts'][0]['text']
                
                pred_is_c4, reasoning = parse_model_response(response_text)
                
                results.append({
                    "sclk": row['sclk'],
                    "true_label": str(row['class']),
                    "pred_is_c4": pred_is_c4,
                    "reasoning": reasoning
                })
                
            except Exception as e:
                print(f"Error processing prediction line {line_no}: {e}")

    if not results:
        print("No results matched!")
        return

    # Convert to DF
    res_df = pd.DataFrame(results)
    
    # Metrics
    y_true = (res_df['true_label'] == '4').astype(int)
    y_pred = res_df['pred_is_c4'].astype(int)
    
    print("\n--- Final Evaluation Metrics ---")
    print(f"Total Predictions Evaluated: {len(res_df)}")
    if len(res_df) < len(df):
        print(f"Warning: Evaluated {len(res_df)} out of {len(df)} samples.")
        
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy: {acc:.2%}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")

    # --- EXPERIMENT TRACKING LOGGING ---
    try:
        if os.path.exists("last_run_config.json"):
            from google.cloud import aiplatform
            from dotenv import load_dotenv
            load_dotenv()
            
            with open("last_run_config.json", "r") as f:
                run_config = json.load(f)
            
            run_name = run_config.get("run_name")
            experiment_name = run_config.get("experiment_name")
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = "us-central1"

            if run_name and experiment_name and project_id:
                print(f"\nLogging metrics to Vertex AI Experiment Run: {run_name}...")
                aiplatform.init(project=project_id, location=location, experiment=experiment_name)
                
                with aiplatform.start_run(run_name, resume=True) as run:
                    metrics = {
                        "accuracy": acc,
                        "recall": rec,
                        "precision": prec,
                        "f1_score": f1,
                        "true_positives": int(tp),
                        "false_negatives": int(fn),
                        "false_positives": int(fp),
                        "true_negatives": int(tn)
                    }
                    run.log_metrics(metrics)
                    print("Metrics logged successfully.")
            else:
                print("Run config found but missing details. Skipping experiment logging.")
    except Exception as e:
        print(f"Warning: Failed to log metrics to experiment: {e}") 
    # --- EXPERIMENT TRACKING ENDS ---

    res_df.to_csv("batch_analysis_results.csv", index=False)
    print("\nDetailed results saved to 'batch_analysis_results.csv'")
    
    fps = res_df[(y_true == 0) & (y_pred == 1)]
    fns = res_df[(y_true == 1) & (y_pred == 0)]
    
    if not fps.empty:
        print(f"\nExample False Positive Reasoning (Sample {fps.iloc[0]['sclk']}):\n{fps.iloc[0]['reasoning']}")
    if not fns.empty:
        print(f"\nExample False Negative Reasoning (Sample {fns.iloc[0]['sclk']}):\n{fns.iloc[0]['reasoning']}")

if __name__ == "__main__":
    main()
