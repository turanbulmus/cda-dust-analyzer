import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

all_classes = ['Noise', '1', '2', '3', '4', '5', '5-Na', '3-P']
display_labels = ['Noise', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 5-Na', 'Class 3-P']

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

def process_run(csv_path, run_name):
    print(f"Processing {run_name} from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    y_true = [normalize_class_label(c) for c in df['true_class']]
    y_pred = [normalize_class_label(c) for c in df['predicted_class']]
    
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
    
    # Save CSVs
    out_dir_study = "study_results_multi_agent"
    out_dir_results = "cda_dust_agent/data/results"
    os.makedirs(out_dir_study, exist_ok=True)
    os.makedirs(out_dir_results, exist_ok=True)
    
    csv_study_path = os.path.join(out_dir_study, f"confusion_matrix_{run_name}.csv")
    csv_results_path = os.path.join(out_dir_results, f"confusion_matrix_{run_name}.csv")
    cm_df.to_csv(csv_study_path)
    cm_df.to_csv(csv_results_path)
    print(f"Saved CSV confusion matrix to {csv_study_path} and {csv_results_path}")
    
    # Render PNG heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=0.5, linecolor='gray')
    plt.title(f"Cassini CDA Mass Spectrum Classification - {run_name.upper()} RUN\n(Confusion Matrix)", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class (Ground Truth)", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    png_study_path = os.path.join(out_dir_study, f"confusion_matrix_{run_name}.png")
    png_results_path = os.path.join(out_dir_results, f"confusion_matrix_{run_name}.png")
    plt.savefig(png_study_path, dpi=300)
    plt.savefig(png_results_path, dpi=300)
    plt.close()
    print(f"Saved PNG confusion matrix plot to {png_study_path} and {png_results_path}")

process_run("cda_dust_agent/data/results/results_2000.csv", "2000")
process_run("cda_dust_agent/data/results/results_1000.csv", "1000")

print("\nAll confusion matrices generated and saved successfully!")
