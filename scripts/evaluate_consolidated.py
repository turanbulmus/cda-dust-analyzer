import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

def evaluate(exclude_noise=False):
    results_csv = "cda_dust_agent/data/results/inf_results.csv"
    if not os.path.exists(results_csv):
        print(f"Error: {results_csv} not found.")
        return

    df = pd.read_csv(results_csv)
    
    # Consolidate labels (e.g., 3-Cl -> 3)
    df['true_consolidated'] = df['true_class'].apply(lambda x: str(x).split('-')[0] if pd.notna(x) else 'Unknown')
    df['predicted_consolidated'] = df['predicted_class'].apply(lambda x: str(x).split('-')[0] if pd.notna(x) else 'Unknown')
    
    if exclude_noise:
        df = df[~df['true_consolidated'].isin(['Noise', '?'])]
        df = df[~df['predicted_consolidated'].isin(['Noise', '?'])]
        print("\n=== Evaluation (Excluding Noise and '?') ===")
    else:
        print("\n=== Evaluation (Including All Classes) ===")
        
    y_true = df['true_consolidated']
    y_pred = df['predicted_consolidated']
    
    target_names = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    
    print(f"Classes: {target_names}")
    
    report = classification_report(y_true, y_pred, labels=target_names)
    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    
    print(report)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    print("Running evaluation with consolidated labels...")
    evaluate(exclude_noise=False)
    evaluate(exclude_noise=True)
