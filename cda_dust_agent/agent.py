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
from .tools.prompt_refinery import run_prompt_refinery

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()
SHARED_STATE = {}

class DataFetchAndParseAgent(BaseAgent):
    """Fetches raw spectra from HuggingFace, filtering, cropping, log-scaling and SAVGOL smoothing it into 4 datasets."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path not in ["local", "batch"]:
            return

        train_L_out = "cda_dust_agent/data/raw/cda_train_L.parquet"
        train_H_out = "cda_dust_agent/data/raw/cda_train_H.parquet"
        inf_L_out = "cda_dust_agent/data/raw/cda_inf_L.parquet"
        inf_H_out = "cda_dust_agent/data/raw/cda_inf_H.parquet"
        
        required_paths = [train_L_out, train_H_out, inf_L_out, inf_H_out]
        
        if not configs.agent_settings.fetch_data:
            if all(os.path.exists(p) for p in required_paths):
                yield log_and_yield(self.name, "Data files exist and fetch_data is False. Skipping fetch and parse.")
                return
            else:
                yield log_and_yield(self.name, "Data files do not exist but fetch_data is False. Proceeding anyway.")

        yield log_and_yield(self.name, "Fetching and processing spectra from HuggingFace... (this may take a moment)")
        
        import huggingface_hub
        import pandas as pd
        import numpy as np
        from scipy.signal import savgol_filter
        
        os.makedirs("cda_dust_agent/data/raw", exist_ok=True)
        
        REPO_ID = "CosmicDustGroup/cassini-cda-spectra"
        FILENAME_INF = "data/lvl2/cda_qm_spectra_pre2008277_inf_lvl2.parquet"
        FILENAME_TRAIN = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"
        
        # Download files
        file_inf_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_INF, repo_type="dataset")
        file_train_path = huggingface_hub.hf_hub_download(repo_id=REPO_ID, filename=FILENAME_TRAIN, repo_type="dataset")
        
        train_df = pd.read_parquet(file_train_path)
        inf_df = pd.read_parquet(file_inf_path)
        
        # 1018 filtering
        train_df = train_df[train_df['spectrum'].apply(len) == 1018].copy()
        inf_df = inf_df[inf_df['spectrum'].apply(len) == 1018].copy()
        
        # Crop spectra to index 10 to 640
        def crop_spectrum(spectrum):
            return spectrum[10:641]
            
        train_df['spectrum'] = train_df['spectrum'].apply(crop_spectrum)
        inf_df['spectrum'] = inf_df['spectrum'].apply(crop_spectrum)
        
        # Re-assign labels X -> ? and group class 3 sub-classes
        def map_labels(x):
            if not isinstance(x, str):
                return x
            if "X" in x:
                return "?"
            if x.startswith("3-") or x == "3":
                return "3"
            return x
            
        train_df['class'] = train_df['class'].apply(map_labels)
        
        # Scaling and Smoothing
        def qm_scaling_savgol(spectrum):
            spectrum = np.array(spectrum, dtype=float)
            log_spec = np.log10(spectrum + np.abs(np.min(spectrum)))
            
            finite_mask = np.isfinite(log_spec)
            if np.any(finite_mask):
                min_finite_val = np.min(log_spec[finite_mask])
            else:
                min_finite_val = 0
                
            log_spec = np.nan_to_num(log_spec, neginf=min_finite_val)
            
            spec_min = np.min(log_spec)
            spec_max = np.max(log_spec)
            range_val = spec_max - spec_min
            
            if range_val > 0:
                scaled_spectrum = (log_spec - spec_min) / range_val
            else:
                scaled_spectrum = np.zeros_like(log_spec)
                
            return savgol_filter(scaled_spectrum, 5, 3)

        train_df['spectrum'] = train_df['spectrum'].apply(qm_scaling_savgol)
        inf_df['spectrum'] = inf_df['spectrum'].apply(qm_scaling_savgol)
        
        # Split into L and H sets based on qi_ampl
        # L: > 10 fC (1e-14), H: between 10 fC and 1 pC (1e-14 and 1e-12)
        # Note: L means amplitude > 10 fC.
        train_L = train_df[train_df['qi_ampl'] < 1e-14].copy()
        train_H = train_df[(train_df['qi_ampl'] >= 1e-14) & (train_df['qi_ampl'] < 1e-12)].copy()
        
        inf_L = inf_df[inf_df['qi_ampl'] < 1e-14].copy()
        inf_H = inf_df[(inf_df['qi_ampl'] >= 1e-14) & (inf_df['qi_ampl'] < 1e-12)].copy()
        
        # Class filtering for L: keep "1", "2", "3", "4", "5", "?", "Noise", convert the rest to "?"
        valid_L_classes = {"1", "2", "3", "4", "5", "Noise"}
        train_L['class'] = train_L['class'].apply(lambda x: x if x in valid_L_classes else "?")
        
        train_L.to_parquet(train_L_out)
        train_H.to_parquet(train_H_out)
        inf_L.to_parquet(inf_L_out)
        inf_H.to_parquet(inf_H_out)
        
        yield log_and_yield(self.name, f"Processed, scaled, smoothed, and split data into L and H targets. Saved: {train_L_out}, {train_H_out}, {inf_L_out}, {inf_H_out}.")

class DataSplitAgent(BaseAgent):
    """Dynamically loads L or H parquet files and splits them into few-shot and validation datasets."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path not in ["local", "batch"]:
            return
            
        import pandas as pd
        import os
        
        # Pull chosen dataset type (L or H) from config. Default to 'L'.
        target_dataset = getattr(configs.agent_settings, "target_dataset", "L")
        train_source = f"cda_dust_agent/data/raw/cda_train_{target_dataset}.parquet"
        
        if not os.path.exists(train_source):
            yield log_and_yield(self.name, f"Training source '{train_source}' not found. Cannot perform split.")
            return
            
        yield log_and_yield(self.name, f"Loading scaled {target_dataset}-dataset from {train_source} for dynamic splitting...")
        df = pd.read_parquet(train_source)
        
        few_shot_split_rule = getattr(configs.agent_settings, "few_shot_split_rules", configs.agent_settings.few_shot_n)
        
        few_shot_list = []
        val_list = []
        
        for cls, group in df.groupby('class'):
            # Determine number of few-shot per class
            n_few_shot = few_shot_split_rule.get(cls, 0) if isinstance(few_shot_split_rule, dict) else int(few_shot_split_rule)
            
            # Shuffle for random split
            group = group.sample(frac=1, random_state=42).reset_index(drop=True)
            
            few_shot_list.append(group.head(n_few_shot))
            val_list.append(group.tail(len(group) - n_few_shot))
            
        few_shot_df = pd.concat(few_shot_list, ignore_index=True) if few_shot_list else pd.DataFrame()
        val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
        
        # Use test_n for capping validation set if in test_mode
        test_n = getattr(configs.agent_settings, "test_n", None) if getattr(configs.agent_settings, "test_mode", False) else None
        if test_n is not None and not val_df.empty:
            val_df = val_df.groupby('class').head(test_n).reset_index(drop=True)
        
        os.makedirs("cda_dust_agent/data/testing", exist_ok=True)
        train_out_path = "cda_dust_agent/data/raw/cda_train.parquet"
        test_out_path = "cda_dust_agent/data/testing/cda_test.parquet"
        
        few_shot_df.to_parquet(train_out_path)
        val_df.to_parquet(test_out_path)
        
        yield log_and_yield(self.name, f"Data splitted dynamically. Few-shot ({len(few_shot_df)}) -> {train_out_path}, Validation ({len(val_df)}) -> {test_out_path}")



class FewShotAnnotationAgent(BaseAgent):
    """Automatically generates few-shot explanations using Gemini."""
    data_path: str = "cda_dust_agent/data/raw/cda_train.parquet"
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        import os
        import json
        if configs.agent_settings.inference_path not in ["local", "batch"]:
            return
            
        if SHARED_STATE.get("fsa_state") == "done":
            return
            
        fsa_state = "auto_annotate"
        
        # Initialize
        os.makedirs("cda_dust_agent/data/input/examples", exist_ok=True)
        cache_file = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
        
        use_cache = not configs.agent_settings.force_new_annotations
        
        if use_cache and os.path.exists(cache_file):
            import json
            import base64
            # Verify cache has items
            with open(cache_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    cached_ex = []
                    for line in lines:
                        if line.strip():
                            entry = json.loads(line)
                            if "image_base64" in entry:
                                entry["image"] = base64.b64decode(entry["image_base64"])
                            cached_ex.append(entry)
                    
                    SHARED_STATE["few_shot_examples"] = cached_ex
                    SHARED_STATE["used_ids"] = [ex["sclk"] for ex in cached_ex]
                    SHARED_STATE["fsa_state"] = "done"
                    yield log_and_yield(self.name, f"Loaded {len(cached_ex)} cached examples. Proceeding to inference...")
                    return
        
        # Clear the cache file since we are intentionally ignoring it or it's empty
        if not use_cache and os.path.exists(cache_file):
            os.remove(cache_file)
        
        yield log_and_yield(self.name, "Starting automatic few-shot annotation with Gemini...")
            
        if fsa_state == "auto_annotate":
            n_per_class = configs.agent_settings.few_shot_n
            df = pd.read_parquet(self.data_path)
            from .tools.utils import get_few_shot_candidates, generate_spectrum_image_bytes
            candidates = get_few_shot_candidates(df, n_per_class=n_per_class)
            
            if not candidates:
                 SHARED_STATE["fsa_state"] = "done"
                 return
                 
            SHARED_STATE["few_shot_examples"] = []
            
            from google import genai
            from google.genai import types
            import base64
            import numpy as np
            
            client = genai.Client()
            model_id = configs.agent_settings.model
            
            # Group candidates by label
            candidates_by_label = {}
            for cand in candidates:
                lbl = cand["label"]
                if lbl not in candidates_by_label:
                    candidates_by_label[lbl] = []
                candidates_by_label[lbl].append(cand)
                
            # Pre-render all candidate images and cache them
            image_cache = {}
            for cand in candidates:
                img_ref_bytes = generate_spectrum_image_bytes(
                    np.array(cand["spectrum"]),
                    title=f"{cand['label']} Reference Sample {cand['sclk']}"
                )
                image_cache[cand['sclk']] = img_ref_bytes

            yield log_and_yield(self.name, f"Generating explanations for {len(candidates)} examples ({n_per_class} per class) using {model_id} with dynamic contrastive references...")
            
            os.makedirs("cda_dust_agent/data/input/examples", exist_ok=True)
            os.makedirs("cda_dust_agent/data/annotated_spectra", exist_ok=True)
            os.makedirs("cda_dust_agent/data/input/store", exist_ok=True)
            cache_file = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
            
            import random
            random.seed(123)
            
            for i, cand in enumerate(candidates):
                img_bytes = generate_spectrum_image_bytes(np.array(cand["spectrum"]), title=f"{cand['label']} Sample {cand['sclk']}")
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                target_label = cand["label"]
                references = []
                for other_label, other_cands in candidates_by_label.items():
                    if other_label != target_label:
                        other_cand = random.choice(other_cands)
                        references.append((other_label, image_cache[other_cand['sclk']]))
                        
                from .prompts import CONTRASTIVE_ANNOTATION_USER_PROMPT, SYSTEM_INSTRUCTION_TEXT
                
                ref_descriptions = ""
                for idx, (other_label, _) in enumerate(references):
                    char_code = chr(66 + idx) # B, C, D, E...
                    ref_descriptions += f"- Image {char_code}: Reference Spectrum of '{other_label}'\n"
                    
                prompt = CONTRASTIVE_ANNOTATION_USER_PROMPT.format(
                    label=target_label,
                    reference_descriptions=ref_descriptions.strip()
                )
                
                parts = [
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png") # Image A
                ]
                for _, other_img_bytes in references:
                    parts.append(types.Part.from_bytes(data=other_img_bytes, mime_type="image/png"))
                    
                contents = [types.Content(role="user", parts=parts)]
                
                try:
                    response = await client.aio.models.generate_content(
                        model=model_id,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION_TEXT
                        )
                    )
                    explanation = response.text.strip()
                except Exception as e:
                    explanation = f"Failed to generate explanation: {e}"
                    
                yield log_and_yield(self.name, f"Annotated {i+1}/{len(candidates)}: {cand['label']} (sclk: {cand['sclk']})")
                
                new_example = {
                    "label": cand["label"],
                    "image": img_bytes,
                    "explanation": explanation,
                    "sclk": cand["sclk"]
                }
                SHARED_STATE["few_shot_examples"].append(new_example)
                
                # Save the annotated spectrum as a PNG file
                import re
                safe_class = re.sub(r'[^\w\s-]', '', cand['label']).replace(' ', '_')
                png_filename = f"cda_dust_agent/data/annotated_spectra/{cand['sclk']}_{safe_class}.png"
                with open(png_filename, "wb") as f:
                    f.write(img_bytes)
                
                # ALSO store in data/input/store
                store_png_filename = f"cda_dust_agent/data/input/store/{cand['sclk']}_{safe_class}.png"
                with open(store_png_filename, "wb") as f:
                    f.write(img_bytes)
                    
                store_json_filename = f"cda_dust_agent/data/input/store/{cand['sclk']}_{safe_class}.json"
                with open(store_json_filename, "w") as f:
                    json.dump({
                        "sclk": cand["sclk"],
                        "label": cand["label"],
                        "prompt": prompt,
                        "explanation": explanation
                    }, f, indent=2)
                
                save_ex = new_example.copy()
                save_ex["image_base64"] = img_b64
                del save_ex["image"]
                
                with open(cache_file, "a") as f:
                    f.write(json.dumps(save_ex) + "\n")
                    
            SHARED_STATE["fsa_state"] = "done"
            SHARED_STATE["used_ids"] = [ex["sclk"] for ex in SHARED_STATE["few_shot_examples"]]
            yield log_and_yield(self.name, f"Automatic few-shot annotations complete. Sent {len(candidates)} spectra to Gemini for few-shot learning. Saved to cache.")
            return

class PromptOptimizerAgent(BaseAgent):
    """Runs the prompt optimization study to improve prompts."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if not configs.agent_settings.prompt_optimization:
            yield log_and_yield(self.name, "Prompt optimization is disabled. Skipping.")
            return
            
        yield log_and_yield(self.name, "Starting prompt optimization study...")
        try:
            system_prompt = await run_prompt_refinery(iterations=configs.agent_settings.prompt_opt_iterations)
            yield log_and_yield(self.name, "Prompt optimization complete. Refined prompt generated.")
            # We could update the system instruction here if needed, or just let the user use the file.
            # For now, we just log it and save it to file (done in prompt_refinery).
        except Exception as e:
            yield log_and_yield(self.name, f"Prompt optimization failed: {e}")
            
class DataPrepAgent(BaseAgent):
    """Reads parquet data, extracts few-shot examples, and generates JSONL."""
    bucket_name: str
    data_path: str = "cda_dust_agent/data/testing/cda_test.parquet"
    limit: int = 0
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return
            
        yield log_and_yield(self.name, f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        limit_val = self.limit if self.limit > 0 else None
        if limit_val:
            df = df.head(limit_val)
            
        few_shot_examples = SHARED_STATE.get("few_shot_examples", [])
        used_ids = SHARED_STATE.get("used_ids", [])
        
        jsonl_file, _ = create_batch_input_file(df, few_shot_examples=few_shot_examples, limit=limit_val)
        yield log_and_yield(self.name, f"Generated {jsonl_file} with {len(df)} spectra for batch inference (excluding few-shot IDs: {used_ids})")
        
        # Save to shared workflow state
        SHARED_STATE["jsonl_file"] = jsonl_file
        SHARED_STATE["bucket_name"] = self.bucket_name

class LocalInferenceAgent(BaseAgent):
    """Runs local inference using Gemini SDK directly."""
    model_id: str
    data_path: str = "cda_dust_agent/data/testing/cda_test.parquet"
    limit: int = 0
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "local":
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return
            
        limit_val = self.limit if self.limit > 0 else None
        yield log_and_yield(self.name, f"Starting local inference on up to {limit_val if limit_val else 'all'} samples using {self.model_id}...")
        
        from google import genai
        from google.genai import types
        import pandas as pd
        import json
        from .tools.utils import generate_spectrum_image_bytes
        from .prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT
        
        client = genai.Client()
        df = pd.read_parquet(self.data_path)
        
        few_shot_examples = SHARED_STATE.get("few_shot_examples", [])
        used_ids = SHARED_STATE.get("used_ids", [])
        
        few_shot_parts = [types.Part.from_text(text=CLASSIFICATION_USER_PROMPT)]
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
                    "class": types.Schema(type=types.Type.STRING, description="The predicted class label"),
                    "explanation": types.Schema(type=types.Type.STRING, description="Explanation for the prediction")
                },
                required=["id", "class", "explanation"]
            )
        )
        
        df_eval = df[~df['sclk'].isin(used_ids)]
        if limit_val:
            # Use random sampling instead of picking the first N samples
            df_eval = df_eval.sample(n=limit_val, random_state=42) if limit_val < len(df_eval) else df_eval
            
        yield log_and_yield(self.name, f"Prepared {len(df_eval)} spectra for local inference.")
        
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
                
                yield log_and_yield(self.name, f"Requesting prediction for item {i+1}/{len(df_eval)} (sclk: {row['sclk']})...")
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
                yield log_and_yield(self.name, f"Record {i+1} failed: {e}")
        
        import os
        os.makedirs("cda_dust_agent/data/output", exist_ok=True)
        preds_file = "cda_dust_agent/data/output/predictions.jsonl"
        with open(preds_file, "w") as f:
            for p in preds:
                f.write(p + "\n")
                
        SHARED_STATE["job_success"] = True
        yield log_and_yield(self.name, f"Local inference complete. Saved to {preds_file}")
        
class BatchSubmissionAgent(BaseAgent):
    """Uploads the JSONL to GCS and submits the Vertex AI batch prediction job."""
    project_id: str
    model_id: str
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return

        jsonl_file = SHARED_STATE.get("jsonl_file")
        bucket_name = SHARED_STATE.get("bucket_name")
        
        if not jsonl_file or not bucket_name:
            yield log_and_yield(self.name, "Missing state: jsonl_file or bucket_name")
            return
            
        yield log_and_yield(self.name, f"Uploading {jsonl_file} to GCS bucket: {bucket_name}")
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
        
        blob = bucket.blob(f"input/{jsonl_file}")
        blob.upload_from_filename(jsonl_file)
        gcs_source = f"gs://{bucket.name}/input/{jsonl_file}"
        
        yield log_and_yield(self.name, f"Uploaded to {gcs_source}. Submitting Batch Job via curl...")
        
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
            yield log_and_yield(self.name, f"Job Submitted Successfully! Job Name: {job_name}")
            SHARED_STATE["job_name"] = job_name
            SHARED_STATE["access_token"] = access_token
        else:
            yield log_and_yield(self.name, f"Error submitting job: {result.stderr}\\nResponse: {result.stdout}")

        if os.path.exists("batch_request.json"):
            os.remove("batch_request.json")
            
class BatchPollingAgent(BaseAgent):
    """Polls Vertex AI until the job completes."""
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return

        job_name = SHARED_STATE.get("job_name")
        access_token = SHARED_STATE.get("access_token")
        
        if not job_name:
            yield log_and_yield(self.name, "No job_name in state to poll.")
            return
            
        check_url = f"https://aiplatform.googleapis.com/v1/{job_name}"
        yield log_and_yield(self.name, "Polling job status...")
        
        while True:
            check_cmd = [
                "curl", "-s", "-X", "GET",
                check_url,
                "-H", f"Authorization: Bearer {access_token}"
            ]
            check_res = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if check_res.returncode != 0:
                yield log_and_yield(self.name, f"Error checking status: {check_res.stderr}")
                time.sleep(30)
                continue
                
            status_data = json.loads(check_res.stdout)
            state = status_data.get("state", "UNKNOWN")
            
            yield log_and_yield(self.name, f"Job State: {state}")
            
            if state == "JOB_STATE_SUCCEEDED":
                SHARED_STATE["job_success"] = True
                break
            elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"]:
                yield log_and_yield(self.name, f"Job Ended with state: {state}")
                if "error" in status_data:
                    yield log_and_yield(self.name, f"Error Details: {status_data['error']}")
                SHARED_STATE["job_success"] = False
                break
                
            time.sleep(30)
            
class ResultAnalysisAgent(BaseAgent):
    """Downloads prediction results and runs analysis."""
    project_id: str
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        inference_path = configs.agent_settings.inference_path
        if inference_path not in ["local", "batch"]:
            return
            
        if SHARED_STATE.get("fsa_state") != "done":
            return
            
        if not SHARED_STATE.get("job_success"):
            yield log_and_yield(self.name, "Job was not successful; skipping analysis.")
            return
            
        if inference_path != "local":
            bucket_name = SHARED_STATE.get("bucket_name")
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
            
            blobs = list(bucket.list_blobs(prefix="output"))
            prediction_blobs = [b for b in blobs if b.name.endswith(".jsonl") and "prediction" in b.name]
            
            if not prediction_blobs:
                yield log_and_yield(self.name, "Warning: No prediction JSONL files found in output directory.")
                return
                
            prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
            blob_to_download = prediction_blobs[0]
            
            yield log_and_yield(self.name, f"Downloading {blob_to_download.name} to cda_dust_agent/data/output/predictions.jsonl...")
            blob_to_download.download_to_filename("cda_dust_agent/data/output/predictions.jsonl")
        else:
            yield log_and_yield(self.name, "Using local cda_dust_agent/data/output/predictions.jsonl...")
        
        yield log_and_yield(self.name, "Running inline analysis of cda_dust_agent/data/output/predictions.jsonl...")
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
            df = pd.read_parquet("cda_dust_agent/data/testing/cda_test.parquet")
            
            y_true = []
            y_pred = []
            explanations = []
            matched_sclks = []
            # Ensure truth_map keys are strictly integer-strings to avoid float matching mismatch ('123.0' vs '123')
            truth_map = {}
            used_ids = set(SHARED_STATE.get("used_ids") or [])
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
                    # Clean the ID string: Models sometimes return "123.0" instead of "123"
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
            
            # Save results to CSV
            import os
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
            
            # Save confusion matrix plot
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

# The root agent that ADK expects
root_agent = SequentialAgent(
    name=configs.agent_settings.name,
    sub_agents=[
        DataFetchAndParseAgent(name="DataFetchAndParse"),
        DataSplitAgent(name="DataSplit"),
        FewShotAnnotationAgent(name="FewShotAnnotation", data_path="cda_dust_agent/data/raw/cda_train.parquet"),
        DataPrepAgent(name="DataPrep", bucket_name=configs.agent_settings.bucket_name, limit=0),
        LocalInferenceAgent(name="LocalInference", model_id=configs.agent_settings.model),
        BatchSubmissionAgent(name="BatchSubmission", project_id=configs.agent_settings.project_id, model_id=configs.agent_settings.model),
        BatchPollingAgent(name="BatchPolling"),
        ResultAnalysisAgent(name="ResultAnalysis", project_id=configs.agent_settings.project_id)
    ],
    description="Orchestrates the data prep, job submission, polling, and results analysis sequentially."
)
