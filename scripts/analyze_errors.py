import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cda_dust_agent.tools.utils import parse_response

def analyze_errors():
    # 1. Reconstruct the sampled dataframe
    train_path = "cda_dust_agent/data/raw/cda_train.parquet"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return
        
    df = pd.read_parquet(train_path)
    df['class'] = df['class'].apply(lambda x: str(x).split('-')[0] if pd.notna(x) else 'Unknown')
    df_sample = df.sample(n=2000, random_state=42)
    
    truth_map = {str(row['sclk']): str(row['class']) for _, row in df_sample.iterrows()}
    
    # 2. Load predictions
    preds_file = "cda_dust_agent/data/output/consolidated_predictions.jsonl"
    if not os.path.exists(preds_file):
        print(f"Error: {preds_file} not found.")
        return
        
    preds = []
    with open(preds_file, 'r') as f:
        for line in f:
            preds.append(json.loads(line))
            
    # 3. Map and Analyze
    analysis_data = []
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
        raw_label = parsed.get("class") or parsed.get("class_label") or "Noise"
        pred_label = str(raw_label).split('-')[0].strip()
        
        pred_id = parsed.get("id")
        explanation = parsed.get("explanation", "")
        
        if pred_id is not None:
            pred_id_str = str(pred_id)
            if pred_id_str in truth_map:
                analysis_data.append({
                    "sclk": pred_id_str,
                    "true_class": truth_map[pred_id_str],
                    "pred_class": pred_label,
                    "explanation": explanation
                })
                
    analysis_df = pd.DataFrame(analysis_data)
    
    # Find misclassifications
    misclassified = analysis_df[analysis_df['true_class'] != analysis_df['pred_class']]
    
    print(f"Total misclassified: {len(misclassified)} out of {len(analysis_df)}")
    
    # Print examples for specific failure modes
    failure_modes = [
        ("1", "Noise"),
        ("2", "Noise"),
        ("3", "Noise"),
        ("4", "Noise"),
        ("Noise", "1"),
        ("Noise", "2")
    ]
    
    for true_c, pred_c in failure_modes:
        subset = misclassified[(misclassified['true_class'] == true_c) & (misclassified['pred_class'] == pred_c)]
        print(f"\n=== Failure Mode: True={true_c} -> Pred={pred_c} (Count: {len(subset)}) ===")
        if len(subset) > 0:
            # Print up to 2 examples
            for _, row in subset.head(2).iterrows():
                print(f"SCLK: {row['sclk']}")
                print(f"Explanation: {row['explanation']}\n")
        else:
            print("No examples found.")

if __name__ == "__main__":
    analyze_errors()
