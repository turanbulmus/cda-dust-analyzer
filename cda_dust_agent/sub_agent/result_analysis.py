import os
import json
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.cloud import storage
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config

from ..tools.utils import parse_response

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()

class ResultAnalysisAgent(BaseAgent):
    """Downloads prediction results and runs analysis."""
    project_id: str
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        inference_path = configs.agent_settings.inference_path
        if inference_path not in ["local", "batch"]:
            return
            
        if ctx.session.state.get("fsa_state") != "done":
            return
            
        if not ctx.session.state.get("job_success"):
            yield log_and_yield(self.name, "Job was not successful; skipping analysis.")
            return
            
        if inference_path != "local":
            os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
            os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"
            bucket_name = ctx.session.state.get("bucket_name")
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
            
            blobs = list(bucket.list_blobs(prefix="output"))
            prediction_blobs = [b for b in blobs if b.name.endswith(".jsonl") and "prediction" in b.name]
            
            if not prediction_blobs:
                yield log_and_yield(self.name, "Warning: No prediction JSONL files found in output directory.")
                return
                
            prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
            newest_blob = prediction_blobs[0]
            folder_prefix = os.path.dirname(newest_blob.name)
            
            latest_run_blobs = [b for b in prediction_blobs if os.path.dirname(b.name) == folder_prefix]
            yield log_and_yield(self.name, f"Downloading {len(latest_run_blobs)} prediction shard file(s) from {folder_prefix} into predictions.jsonl...")
            
            os.makedirs("cda_dust_agent/data/output", exist_ok=True)
            output_filepath = "cda_dust_agent/data/output/predictions.jsonl"
            with open(output_filepath, 'wb') as outfile:
                for blob in sorted(latest_run_blobs, key=lambda x: x.name):
                    content = blob.download_as_bytes()
                    outfile.write(content)
                    if not content.endswith(b'\n'):
                        outfile.write(b'\n')
            
            # Merge bypassed predictions from Stage 1 Pre-filter if present
            bypassed_file = "cda_dust_agent/data/output/bypassed_predictions.jsonl"
            if os.path.exists(bypassed_file):
                yield log_and_yield(self.name, f"Merging bypassed noise predictions from {bypassed_file} into {output_filepath}...")
                with open(bypassed_file, 'rb') as bf, open(output_filepath, 'ab') as outfile:
                    outfile.write(bf.read())
        else:
            yield log_and_yield(self.name, "Using local cda_dust_agent/data/output/predictions.jsonl...")
            # Also support local mode merge if present
            bypassed_file = "cda_dust_agent/data/output/bypassed_predictions.jsonl"
            output_filepath = "cda_dust_agent/data/output/predictions.jsonl"
            if os.path.exists(bypassed_file):
                yield log_and_yield(self.name, f"Merging bypassed noise predictions from {bypassed_file} into {output_filepath}...")
                # We need to read predictions.jsonl, check if they are already merged, if not merge
                with open(output_filepath, 'r') as f:
                    content = f.read()
                # Check if bypassed SCLKs are already in content to avoid double merging
                with open(bypassed_file, 'r') as bf:
                    first_line = bf.readline()
                if first_line:
                    try:
                        p_first = json.loads(first_line)
                        p_id = parse_response(p_first.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')).get('id')
                        if p_id and p_id not in content:
                            with open(output_filepath, 'ab') as outfile, open(bypassed_file, 'rb') as bf:
                                outfile.write(bf.read())
                    except Exception:
                        pass
        
        yield log_and_yield(self.name, "Running inline analysis of cda_dust_agent/data/output/predictions.jsonl...")
        try:
            from sklearn.metrics import classification_report, confusion_matrix

            # Load predictions
            preds = []
            with open("cda_dust_agent/data/output/predictions.jsonl", 'r') as f:
                for line in f:
                    preds.append(json.loads(line))
            
            # Load ground truth
            df = pd.read_parquet("cda_dust_agent/data/testing/cda_test.parquet")
            
            y_true = []
            y_pred = []
            explanations = []
            matched_sclks = []
            truth_map = {}
            used_ids = set(ctx.session.state.get("used_ids") or [])
            df_eval = df[~df['sclk'].isin(used_ids)]
            for k, v in df_eval.set_index('sclk')['class'].to_dict().items():
                try:
                    clean_k = str(int(float(k)))
                except ValueError:
                    clean_k = str(k)
                truth_map[clean_k] = str(v)
            
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
                    
                raw_label = str(parsed.get("class") or parsed.get("class_label") or "Noise").strip()
                valid_classes = ['Noise', '1', '2', '3', '4', '5', '5-Na', '3-P']
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
                        
                pred_id = parsed.get("id")
                explanation = parsed.get("explanation", "")
                if pred_id is not None:
                    try:
                        clean_id = str(int(float(pred_id)))
                    except ValueError:
                        clean_id = str(pred_id)
                        
                    if clean_id in truth_map:
                        matched_sclks.append(clean_id)
                        y_true.append(truth_map[clean_id])
                        y_pred.append(pred_label)
                        explanations.append(explanation)
                
            yield log_and_yield(self.name, f"Parsed {len(y_true)} matched predictions.")

            target_names = sorted(list({str(v) for v in truth_map.values()}))
            if not target_names:
                target_names = ['4', '1', 'Noise']
                
            report = classification_report(y_true, y_pred, labels=target_names)
            cm = confusion_matrix(y_true, y_pred, labels=target_names)
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            os.makedirs("cda_dust_agent/data/results", exist_ok=True)
            results_df = pd.DataFrame({
                "sclk": matched_sclks,
                "true_class": y_true,
                "predicted_class": y_pred,
                "explanation": explanations
            })
            results_df.to_csv("cda_dust_agent/data/results/results.csv", index=False)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig("cda_dust_agent/data/results/confusion_matrix.png")
            plt.close()
            
            analysis_text = f"\n--- Evaluation Results ---\n{report}\n\nConfusion Matrix (Rows=True, Cols=Pred):\nLabels: {target_names}\n{cm}\nSaved detailed results to cda_dust_agent/data/results/results.csv\nSaved confusion matrix plot to cda_dust_agent/data/results/confusion_matrix.png\n"
            yield log_and_yield(self.name, analysis_text)

        except Exception as e:
            yield log_and_yield(self.name, f"Error running analysis: {e}")
