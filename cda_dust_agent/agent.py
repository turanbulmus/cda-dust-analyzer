import os
import json
import time
import subprocess
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.cloud import storage

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from .config import Config
from .tools.utils import create_batch_input_file

configs = Config()
SHARED_STATE = {}

class RoutingAgent(BaseAgent):
    """Prompts the user to choose between local or batch inference."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if SHARED_STATE.get("inference_path") in ["local", "batch"]:
            return

        choice = None
        if ctx.user_content and ctx.user_content.parts:
            text = ctx.user_content.parts[0].text.strip().lower()
            if text in ["local", "batch"]:
                choice = text
            elif "local" in text:
                choice = "local"
            elif "batch" in text:
                choice = "batch"

        if not choice:
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text="Run (local) inference or (batch) inference? [local/batch]: ")]))
            return

        SHARED_STATE["inference_path"] = choice
        yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Selected inference path: {choice}")]))

class FewShotAnnotationAgent(BaseAgent):
    """Interactively prompts user for few-shot explanations."""
    data_path: str = "cda_dust_agent/data/raw/cda_sample.parquet"
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if SHARED_STATE.get("inference_path") not in ["local", "batch"]:
            return
            
        if SHARED_STATE.get("fsa_state") == "done":
            return
            
        fsa_state = SHARED_STATE.get("fsa_state")
        
        # Initialize
        if not fsa_state:
            SHARED_STATE["fsa_state"] = "ask_n"
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text="How many few-shot examples per class should we use? [default: 2]: ")]))
            return
            
        if fsa_state == "ask_n":
            n_per_class = 2
            if ctx.user_content and ctx.user_content.parts:
                text = ctx.user_content.parts[0].text.strip()
                if text.isdigit():
                    n_per_class = int(text)
            SHARED_STATE["n_per_class"] = n_per_class
            SHARED_STATE["fsa_state"] = "annotating"
            SHARED_STATE["fsa_current_idx"] = 0
            SHARED_STATE["few_shot_examples"] = []
            
            df = pd.read_parquet(self.data_path)
            from .tools.utils import get_few_shot_candidates
            candidates = get_few_shot_candidates(df, n_per_class=n_per_class)
            SHARED_STATE["fsa_candidates"] = candidates
            
            if not candidates:
                 SHARED_STATE["fsa_state"] = "done"
                 return
                 
            # Prompt for first candidate
            first_cand = candidates[0]
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.semilogy(first_cand["spectrum"], color='black', linewidth=1.5)
            plt.axvspan(180, 400, color='green', alpha=0.1, label='Class 4 Region')
            plt.axvspan(0, 50, color='red', alpha=0.1, label='Noise Region')
            plt.title(f"{first_cand['label']} Sample {first_cand['sclk']}")
            plt.grid(True)
            plt.show(block=False)
            plt.pause(0.1)
            
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Please provide an explanation for {first_cand['label']} (sclk: {first_cand['sclk']}): ")]))
            return
            
        if fsa_state == "annotating":
            candidates = SHARED_STATE["fsa_candidates"]
            current_idx = SHARED_STATE["fsa_current_idx"]
            
            explanation = "No explanation provided."
            if ctx.user_content and ctx.user_content.parts:
                text = ctx.user_content.parts[0].text.strip()
                if text:
                    explanation = text
                
            current_cand = candidates[current_idx]
            from .tools.utils import generate_spectrum_image_bytes
            import numpy as np
            
            img_bytes = generate_spectrum_image_bytes(np.array(current_cand["spectrum"]), title=f"{current_cand['label']} Sample {current_cand['sclk']}")
            
            SHARED_STATE["few_shot_examples"].append({
                "label": current_cand["label"],
                "image": img_bytes,
                "explanation": explanation,
                "sclk": current_cand["sclk"]
            })
            
            import matplotlib.pyplot as plt
            if plt.fignum_exists(plt.gcf().number):
                plt.close() # close the current plot
            
            next_idx = current_idx + 1
            if next_idx < len(candidates):
                SHARED_STATE["fsa_current_idx"] = next_idx
                next_cand = candidates[next_idx]
                
                plt.figure(figsize=(12, 6))
                plt.semilogy(next_cand["spectrum"], color='black', linewidth=1.5)
                plt.axvspan(180, 400, color='green', alpha=0.1, label='Class 4 Region')
                plt.axvspan(0, 50, color='red', alpha=0.1, label='Noise Region')
                plt.title(f"{next_cand['label']} Sample {next_cand['sclk']}")
                plt.grid(True)
                plt.show(block=False)
                plt.pause(0.1)
                
                yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Please provide an explanation for {next_cand['label']} (sclk: {next_cand['sclk']}): ")]))
                return
            else:
                SHARED_STATE["fsa_state"] = "done"
                yield Event(author=self.name, content=Content(parts=[Part.from_text(text="Few-shot annotations complete.")]))
                SHARED_STATE["used_ids"] = [ex["sclk"] for ex in SHARED_STATE["few_shot_examples"]]
                return

class DataPrepAgent(BaseAgent):
    """Reads parquet data, extracts few-shot examples, and generates JSONL."""
    bucket_name: str
    data_path: str = "cda_dust_agent/data/raw/cda_sample.parquet"
    limit: int = 0
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if SHARED_STATE.get("inference_path") != "batch":
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return
            
        yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Loading data from {self.data_path}")]))
        df = pd.read_parquet(self.data_path)
        
        limit_val = self.limit if self.limit > 0 else None
        if limit_val:
            df = df.head(limit_val)
            
        few_shot_examples = SHARED_STATE.get("few_shot_examples", [])
        used_ids = SHARED_STATE.get("used_ids", [])
        
        jsonl_file, _ = create_batch_input_file(df, few_shot_examples=few_shot_examples, limit=limit_val)
        yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Generated {jsonl_file} excluding few-shot IDs: {used_ids}")]))
        
        # Save to shared workflow state
        SHARED_STATE["jsonl_file"] = jsonl_file
        SHARED_STATE["bucket_name"] = self.bucket_name

class LocalInferenceAgent(BaseAgent):
    """Runs local inference using Gemini SDK directly."""
    model_id: str
    data_path: str = "cda_dust_agent/data/raw/cda_sample.parquet"
    limit: int = 5
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if SHARED_STATE.get("inference_path") != "local":
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return
            
        yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Starting local inference on up to {self.limit} samples using {self.model_id}...")]))
        
        from google import genai
        from google.genai import types
        import pandas as pd
        import json
        from .tools.utils import generate_spectrum_image_bytes
        from .prompts import SYSTEM_INSTRUCTION_TEXT, USER_PROMPT_TEXT
        
        client = genai.Client()
        df = pd.read_parquet(self.data_path)
        
        few_shot_examples = SHARED_STATE.get("few_shot_examples", [])
        used_ids = SHARED_STATE.get("used_ids", [])
        
        few_shot_parts = [types.Part.from_text(text=USER_PROMPT_TEXT)]
        if few_shot_examples:
            few_shot_parts.append(types.Part.from_text(text="Here are reference examples:"))
            for ex in few_shot_examples:
                few_shot_parts.append(types.Part.from_text(text=f"Example: {ex['label']} ({ex['explanation']})"))
                few_shot_parts.append(types.Part.from_bytes(data=ex['image'], mime_type="image/png"))
            few_shot_parts.append(types.Part.from_text(text="Now, analyze the following spectrum:"))
            
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION_TEXT,
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "id": types.Schema(type=types.Type.STRING, description="The ID of the run (sclk)"),
                    "class": types.Schema(type=types.Type.STRING, description="The predicted class ('4', '1', or 'Noise')"),
                    "explanation": types.Schema(type=types.Type.STRING, description="Explanation for the prediction")
                },
                required=["id", "class", "explanation"]
            )
        )
        
        df_eval = df[~df['sclk'].isin(used_ids)]
        limit_val = self.limit if self.limit > 0 else len(df_eval)
        # Use random sampling instead of picking the first N samples
        df_eval = df_eval.sample(n=limit_val, random_state=42) if limit_val < len(df_eval) else df_eval
        
        preds = []
        
        import os
        import base64
        import json
        
        os.makedirs("cda_dust_agent/data/input", exist_ok=True)
        local_reqs_file = "cda_dust_agent/data/input/local_requests.jsonl"
        with open(local_reqs_file, "w") as f:
            pass # clear the file
            
        for i, (_, row) in enumerate(df_eval.iterrows()):
            try:
                spect = row['spectrum']
                img_bytes = generate_spectrum_image_bytes(spect, title=f"Sample {row['sclk']}")
                
                parts = list(few_shot_parts)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
                parts.append(types.Part.from_text(text=f"Sample ID (sclk): {row['sclk']}"))
                
                contents = [types.Content(role="user", parts=parts)]
                
                req_dict = {
                    "request": {
                        "contents": [{"role": "user", "parts": []}],
                        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION_TEXT}]},
                    }
                }
                for p in parts:
                    if p.text:
                        req_dict["request"]["contents"][0]["parts"].append({"text": p.text})
                    elif p.inline_data:
                        req_dict["request"]["contents"][0]["parts"].append({
                            "inline_data": {
                                "mime_type": p.inline_data.mime_type,
                                "data": base64.b64encode(p.inline_data.data).decode('utf-8')
                            }
                        })
                with open(local_reqs_file, "a") as f:
                    f.write(json.dumps(req_dict) + "\n")
                
                yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Requesting prediction for item {i+1}/{limit_val}...")]))
                response = await client.aio.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config
                )
                
                # Format to match batch prediction output shape for analysis agent
                pred_record = {
                    "request": {},
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": response.text}]
                                }
                            }
                        ]
                    }
                }
                preds.append(json.dumps(pred_record))
                
            except Exception as e:
                yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Record {i+1} failed: {e}")]))
        
        import os
        os.makedirs("cda_dust_agent/data/output", exist_ok=True)
        preds_file = "cda_dust_agent/data/output/predictions.jsonl"
        with open(preds_file, "w") as f:
            for p in preds:
                f.write(p + "\n")
                
        SHARED_STATE["job_success"] = True
        yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Local inference complete. Saved to {preds_file}")]))
        
class BatchSubmissionAgent(BaseAgent):
    """Uploads the JSONL to GCS and submits the Vertex AI batch prediction job."""
    project_id: str
    model_id: str
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if SHARED_STATE.get("inference_path") != "batch":
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return

        jsonl_file = SHARED_STATE.get("jsonl_file")
        bucket_name = SHARED_STATE.get("bucket_name")
        
        if not jsonl_file or not bucket_name:
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text="Missing state: jsonl_file or bucket_name")]))
            return
            
        yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Uploading {jsonl_file} to GCS bucket: {bucket_name}")]))
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
        
        blob = bucket.blob(f"input/{jsonl_file}")
        blob.upload_from_filename(jsonl_file)
        gcs_source = f"gs://{bucket.name}/input/{jsonl_file}"
        
        yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Uploaded to {gcs_source}. Submitting Batch Job via curl...")]))
        
        # Fetching auth token
        access_token = os.popen("gcloud auth application-default print-access-token").read().strip()
        job_display_name = f"cda-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"    
        global_batch_req = {
            "displayName": job_display_name,
            "model": f"publishers/google/models/{self.model_id}",
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": [gcs_source]}
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": f"gs://{bucket.name}/output"}
            }
        }
        
        with open("batch_request.json", "w") as f:
            json.dump(global_batch_req, f)
            
        curl_command = [
            "curl", "-s", "-X", "POST",
            f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/global/batchPredictionJobs",
            "-H", f"Authorization: Bearer {access_token}",
            "-H", "Content-Type: application/json; charset=utf-8",
            "-d", "@batch_request.json"
        ]
        
        result = subprocess.run(curl_command, capture_output=True, text=True)
        if result.returncode == 0 and "name" in result.stdout:
            response_json = json.loads(result.stdout)
            job_name = response_json.get("name")
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Job Submitted Successfully! Job Name: {job_name}")]))
            SHARED_STATE["job_name"] = job_name
            SHARED_STATE["access_token"] = access_token
        else:
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Error submitting job: {result.stderr}\\nResponse: {result.stdout}")]))

        if os.path.exists("batch_request.json"):
            os.remove("batch_request.json")
            
class BatchPollingAgent(BaseAgent):
    """Polls Vertex AI until the job completes."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if SHARED_STATE.get("inference_path") != "batch":
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return

        job_name = SHARED_STATE.get("job_name")
        access_token = SHARED_STATE.get("access_token")
        
        if not job_name:
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text="No job_name in state to poll.")]))
            return
            
        check_url = f"https://aiplatform.googleapis.com/v1/{job_name}"
        yield Event(author=self.name, content=Content(parts=[Part.from_text(text="Polling job status...")]))
        
        while True:
            check_cmd = [
                "curl", "-s", "-X", "GET",
                check_url,
                "-H", f"Authorization: Bearer {access_token}"
            ]
            check_res = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if check_res.returncode != 0:
                yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Error checking status: {check_res.stderr}")]))
                time.sleep(30)
                continue
                
            status_data = json.loads(check_res.stdout)
            state = status_data.get("state", "UNKNOWN")
            
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Job State: {state}")]))
            
            if state == "JOB_STATE_SUCCEEDED":
                SHARED_STATE["job_success"] = True
                break
            elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
                yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Job Ended with state: {state}")]))
                if "error" in status_data:
                    yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Error Details: {status_data['error']}")]))
                SHARED_STATE["job_success"] = False
                break
                
            time.sleep(30)
            
class ResultAnalysisAgent(BaseAgent):
    """Downloads prediction results and runs analysis."""
    project_id: str
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        inference_path = SHARED_STATE.get("inference_path")
        if inference_path not in ["local", "batch"]:
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return
            
        if not SHARED_STATE.get("job_success"):
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text="Job was not successful; skipping analysis.")]))
            return
            
        if inference_path != "local":
            bucket_name = SHARED_STATE.get("bucket_name")
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
            
            blobs = list(bucket.list_blobs(prefix="output"))
            prediction_blobs = [b for b in blobs if b.name.endswith(".jsonl") and "prediction" in b.name]
            
            if not prediction_blobs:
                yield Event(author=self.name, content=Content(parts=[Part.from_text(text="Warning: No prediction JSONL files found in output directory.")]))
                return
                
            prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
            blob_to_download = prediction_blobs[0]
            
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Downloading {blob_to_download.name} to cda_dust_agent/data/output/predictions.jsonl...")]))
            blob_to_download.download_to_filename("cda_dust_agent/data/output/predictions.jsonl")
        else:
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text="Using local cda_dust_agent/data/output/predictions.jsonl...")]))
        
        yield Event(author=self.name, content=Content(parts=[Part.from_text(text="Running inline analysis of cda_dust_agent/data/output/predictions.jsonl...")]))
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            from .tools.utils import parse_response

            # Load predictions
            preds = []
            with open("cda_dust_agent/data/output/predictions.jsonl", 'r') as f:
                for line in f:
                    preds.append(json.loads(line))
            
            # Load ground truth
            # We assume it matches the original data parsed in DataPrepAgent
            df = pd.read_parquet("cda_dust_agent/data/raw/cda_sample.parquet")
            
            y_true = []
            y_pred = []
            explanations = []
            matched_sclks = []
            
            used_ids = set(SHARED_STATE.get("used_ids", []))
            df_eval = df[~df['sclk'].isin(used_ids)]
            truth_map = {str(k): str(v) for k, v in df_eval.set_index('sclk')['class'].to_dict().items()}
            
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
                raw_label_str = str(raw_label).strip().lower()
                
                # Dynamic matching based on truth variables
                unique_classes_lower = {str(k).lower(): str(k) for k in truth_map.values()}
                pred_label = "Noise"
                for cls_lower, cls_real in unique_classes_lower.items():
                    if cls_lower in raw_label_str:
                        pred_label = cls_real
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
                
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Parsed {len(y_true)} matched predictions.")]))

            target_names = sorted(list({str(v) for v in truth_map.values()}))
            if not target_names:
                target_names = ['4', '1', 'Noise']
                
            report = classification_report(y_true, y_pred, labels=target_names)
            cm = confusion_matrix(y_true, y_pred, labels=target_names)
            
            # Save results to CSV
            import os
            os.makedirs("cda_dust_agent/data/results", exist_ok=True)
            results_df = pd.DataFrame({
                "sclk": matched_sclks,
                "true_class": y_true,
                "predicted_class": y_pred,
                "explanation": explanations
            })
            results_df.to_csv("cda_dust_agent/data/results/results.csv", index=False)
            
            analysis_text = f"\n--- Evaluation Results ---\n{report}\n\nConfusion Matrix (Rows=True, Cols=Pred):\nLabels: {target_names}\n{cm}\nSaved detailed results to cda_dust_agent/data/results/results.csv\n"
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text=analysis_text)]))

        except Exception as e:
            yield Event(author=self.name, content=Content(parts=[Part.from_text(text=f"Error running analysis: {e}")]))

# The root agent that ADK expects
root_agent = SequentialAgent(
    name=configs.agent_settings.name,
    sub_agents=[
        RoutingAgent(name="Routing"),
        FewShotAnnotationAgent(name="FewShotAnnotation", data_path="cda_dust_agent/data/raw/cda_sample.parquet"),
        DataPrepAgent(name="DataPrep", bucket_name=configs.agent_settings.bucket_name, limit=0),
        LocalInferenceAgent(name="LocalInference", model_id=configs.agent_settings.model),
        BatchSubmissionAgent(name="BatchSubmission", project_id=configs.agent_settings.project_id, model_id=configs.agent_settings.model),
        BatchPollingAgent(name="BatchPolling"),
        ResultAnalysisAgent(name="ResultAnalysis", project_id=configs.agent_settings.project_id)
    ],
    description="Orchestrates the data prep, job submission, polling, and results analysis sequentially."
)
