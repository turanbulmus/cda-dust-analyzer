import os
from datetime import datetime
import pandas as pd
from google.cloud import aiplatform
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("=================================================================")
print(" LOGGING 1,000 SAMPLE EXPERIMENT TO VERTEX AI EXPERIMENTS ")
print("=================================================================")

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "turan-genai-bb")
location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"

results_csv = "cda_dust_agent/data/results/results_1000.csv"
if not os.path.exists(results_csv):
    raise FileNotFoundError(f"Results file not found at {results_csv}")

df = pd.read_csv(results_csv)
y_true = df['true_class'].astype(str)
y_pred = df['predicted_class'].astype(str)

accuracy = accuracy_score(y_true, y_pred)
weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

metrics = {
    "accuracy": float(accuracy),
    "weighted_f1": float(weighted_f1),
    "macro_f1": float(macro_f1),
    "noise_recall": float(report_dict.get('Noise', {}).get('recall', 0.0)),
    "noise_precision": float(report_dict.get('Noise', {}).get('precision', 0.0)),
    "noise_f1": float(report_dict.get('Noise', {}).get('f1-score', 0.0)),
    "class1_water_ice_recall": float(report_dict.get('1', {}).get('recall', 0.0)),
    "class2_organic_ice_f1": float(report_dict.get('2', {}).get('f1-score', 0.0)),
    "class4_silicate_recall": float(report_dict.get('4', {}).get('recall', 0.0)),
}

params = {
    "model": "gemini-3.5-flash",
    "inference_path": "batch",
    "few_shot_n": 16,
    "test_n": len(df),
    "prompt_version": "rules_0_to_5_with_qi_metadata",
    "disqualification_rules": "0-NoiseGate,1-GrassVsOrg,2-AlkaliDoublet,3-Orthophosphate,4-Class3PExcl,5-SilicateSpike",
    "qi_ampl_metadata_injected": True
}

experiment_name = "cda-dust-analyzer-experiment"
run_name = f"run-1000samples-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

print(f"Initializing Vertex AI Experiment: {experiment_name}, Run: {run_name}")

aiplatform.init(
    project=project_id,
    location=location,
    experiment=experiment_name
)

with aiplatform.start_run(run=run_name):
    print("Logging parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    aiplatform.log_params(params)
    
    print("\nLogging metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    aiplatform.log_metrics(metrics)

print("\nVertex AI Experiments logging complete successfully!")
