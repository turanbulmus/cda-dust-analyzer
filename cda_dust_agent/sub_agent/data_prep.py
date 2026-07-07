import os
import json
import logging
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from google.cloud import storage
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()


class FilterAndPlotDoFn(beam.DoFn):
    """Processes a single spectrum record, pre-filters Noise, and generates VLM prompt payloads."""
    
    def __init__(self, few_shot_gcs_path, system_instruction, user_prompt, general_profiles, project_id):
        self.few_shot_gcs_path = few_shot_gcs_path
        self.system_instruction = system_instruction
        self.user_prompt = user_prompt
        self.general_profiles = general_profiles
        self.project_id = project_id
        self.few_shot_examples = None

    def setup(self):
        """Loads few-shot examples from GCS once per worker node initialization."""
        import json
        from google.cloud import storage
        
        logger.info(f"Worker setup: Loading few-shot examples from {self.few_shot_gcs_path}...")
        
        # Parse GCS path
        path_parts = self.few_shot_gcs_path.replace("gs://", "").split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()
        
        self.few_shot_examples = []
        for line in content.splitlines():
            if line.strip():
                self.few_shot_examples.append(json.loads(line))
        logger.info(f"Worker setup complete! Loaded {len(self.few_shot_examples)} reference examples.")

    def is_obvious_noise(self, spectrum_data, qi_ampl):
        """Evaluates if the spectrum is flat/empty noise using baseline range and peak detection."""
        import numpy as np
        import pandas as pd
        from scipy.signal import find_peaks
        
        spec = np.array(spectrum_data)
        qi = float(qi_ampl) if qi_ampl is not None else 0.0
        
        # 1. Smoothed range
        smoothed = pd.Series(spec).rolling(window=15, center=True).mean().dropna().values
        if len(smoothed) == 0:
            return True
        smoothed_range = smoothed.max() - smoothed.min()
        
        # 2. Peaks
        inverted = 1.0 - spec
        peaks, _ = find_peaks(inverted, height=0.20, prominence=0.08, width=1)
        valid_peaks = [p for p in peaks if 20 < p < 620]
        
        # Gating decisions:
        is_flat = smoothed_range < 0.075
        no_peaks_low_charge = (len(valid_peaks) == 0) and (qi < 1e-15)
        
        return is_flat or no_peaks_low_charge

    def generate_spectrum_image_bytes(self, spectrum_data, title=None, dpi=60):
        """Plots spectrum and returns PNG bytes."""
        import io
        import matplotlib
        matplotlib.use('Agg') # Headless backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
        ax.plot(spectrum_data, color='black', linewidth=1.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(10, 640)
        ax.set_xlabel("Time-of-Flight Channel Index")
        ax.set_ylabel("Normalized Amplitude")
        if title:
            ax.set_title(title)
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def process(self, record):
        """Processes a single spectrum dict record."""
        import json
        import base64
        
        sclk = int(record["sclk"])
        spectrum = list(record["spectrum"])
        qi_ampl = record.get("qi_ampl", 0.0)
        true_class = record.get("class", "Noise")
        
        sclk_str = str(sclk)
        
        # 1. Apply pre-filter
        if self.is_obvious_noise(spectrum, qi_ampl):
            bypassed_pred = {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps({
                                            "id": sclk_str,
                                            "class": "Noise",
                                            "explanation": "Bypassed by Stage 1 peak detection pre-filter (flat quiet baseline, no physical peaks)."
                                        })
                                    }
                                ],
                                "role": "model"
                            }
                        }
                    ]
                }
            }
            yield beam.pvalue.TaggedOutput("bypassed", json.dumps(bypassed_pred))
            return
            
        # 2. Render plot & encode
        img_bytes = self.generate_spectrum_image_bytes(spectrum, title=f"Sample {sclk_str}")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # 3. Construct visual prompt parts
        contents_parts = [{"text": self.user_prompt}]
        contents_parts.append({"text": "Here are reference examples:"})
        
        for ex in self.few_shot_examples:
            raw_key = ex.get('raw_class', ex['label'].replace("Class ", "").strip())
            gen_profile = self.general_profiles.get(raw_key, "")
            contents_parts.append({
                "text": f"Example: {ex['label']}\n- General Profile: {gen_profile}\n- Specific Sample Features: {ex['explanation']}"
            })
            contents_parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": ex["image_base64"]
                }
            })
            
        contents_parts.append({"text": "Now, analyze the following spectrum:"})
        contents_parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": img_b64
            }
        })
        
        qi_val = str(qi_ampl)
        contents_parts.append({"text": f"Sample ID (sclk): {sclk_str} | Target Charge (qi_ampl): {qi_val} C"})
        
        # 4. Format to Vertex AI Batch Prediction API request schema
        vlm_request = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": contents_parts
                    }
                ],
                "systemInstruction": {
                    "parts": [
                        {
                            "text": self.system_instruction
                        }
                    ]
                }
            }
        }
        
        yield beam.pvalue.TaggedOutput("queued", json.dumps(vlm_request))


class DataPrepAgent(BaseAgent):
    """Pre-filters noise, renders spectrum plots, and generates visual VLM batch requests using Apache Beam (Cloud Dataflow)."""
    bucket_name: str
    data_path: str = "cda_dust_agent/data/testing/cda_test.parquet"
    limit: int = 0
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if configs.agent_settings.inference_path != "batch":
            return
            
        if ctx.session.state.get("fsa_state") != "done":
            return

        # Setup GCS Storage Client
        os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
        os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"
        
        project_id = configs.agent_settings.project_id
        clean_bucket = self.bucket_name.replace("gs://", "").strip("/")
        
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(clean_bucket)
        
        # 1. Clean existing input/bypassed files in GCS
        existing_inputs = [b for b in bucket.list_blobs(prefix="input/batch_requests_part") if b.name.endswith(".jsonl")]
        existing_bypassed = [b for b in bucket.list_blobs(prefix="output/bypassed/bypassed_predictions") if b.name.endswith(".jsonl")]
        
        if existing_inputs or existing_bypassed:
            yield log_and_yield(self.name, f"Cleaning up {len(existing_inputs)} old input files and {len(existing_bypassed)} old bypassed files in GCS...")
            for b in existing_inputs + existing_bypassed:
                b.delete()
            yield log_and_yield(self.name, "GCS input/bypassed folders cleared.")

        # 2. Upload fresh dataset Parquet and cached few-shot examples to GCS
        few_shot_cache_local = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
        
        yield log_and_yield(self.name, "Uploading test Parquet and few-shot annotations to GCS...")
        
        parquet_blob = bucket.blob("data/cda_test.parquet")
        parquet_blob.upload_from_filename(self.data_path)
        
        few_shot_blob = bucket.blob("data/cached_examples.jsonl")
        few_shot_blob.upload_from_filename(few_shot_cache_local)
        
        yield log_and_yield(self.name, "Upload complete! Files staged on GCS.")

        # 3. Determine runner and options
        test_mode = configs.agent_settings.test_mode
        runner = "DirectRunner" if test_mode else "DataflowRunner"
        limit_val = self.limit if self.limit > 0 else (configs.agent_settings.test_n if test_mode else 0)
        
        job_name = f"cda-data-prep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        yield log_and_yield(self.name, f"Initializing Apache Beam pipeline using {runner} inline (Job Name: {job_name}, Limit: {limit_val})...")
        
        # Read dataset parquet locally to feed to the pipeline
        import pandas as pd
        df = pd.read_parquet(self.data_path)
        
        if limit_val > 0 and limit_val < len(df):
            yield log_and_yield(self.name, f"Sampling {limit_val} records for execution...")
            df = df.sample(n=limit_val, random_state=42)
            
        # Convert numpy types to native Python types to avoid serialization/version issues across worker nodes
        records = []
        for _, row in df.iterrows():
            records.append({
                "sclk": int(row["sclk"]),
                "spectrum": [float(v) for v in row["spectrum"]],
                "qi_ampl": float(row["qi_ampl"]) if pd.notna(row.get("qi_ampl", 0.0)) else 0.0,
                "class": str(row.get("class", "Noise"))
            })
            
        yield log_and_yield(self.name, f"Normalized {len(records)} records. Staging pipeline execution...")

        # 4. Construct PipelineOptions
        options_args = [
            f"--runner={runner}",
            f"--project={project_id}",
            f"--temp_location=gs://{clean_bucket}/temp",
            f"--staging_location=gs://{clean_bucket}/staging",
        ]
        
        if runner == "DataflowRunner":
            options_args.extend([
                f"--region={configs.agent_settings.location}",
                "--setup_file=./setup.py",  # Packages cda_dust_agent source code for workers
                f"--job_name={job_name}"
            ])
            
        pipeline_options = PipelineOptions(options_args)
        pipeline_options.view_as(SetupOptions).save_main_session = True
        
        # Load prompt resources to pass to DoFn
        from cda_dust_agent.prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT, GENERAL_PROFILES
        
        few_shot_gcs_path = f"gs://{clean_bucket}/data/cached_examples.jsonl"
        
        # 5. Build and execute pipeline inline
        try:
            with beam.Pipeline(options=pipeline_options) as p:
                dist_records = p | "Create Records" >> beam.Create(records)
                
                results = dist_records | "Filter and Plot" >> beam.ParDo(
                    FilterAndPlotDoFn(
                        few_shot_gcs_path=few_shot_gcs_path,
                        system_instruction=SYSTEM_INSTRUCTION_TEXT,
                        user_prompt=CLASSIFICATION_USER_PROMPT,
                        general_profiles=GENERAL_PROFILES,
                        project_id=project_id
                    )
                ).with_outputs("queued", "bypassed")
                
                results.queued | "Write Queued Requests" >> beam.io.WriteToText(
                    f"gs://{clean_bucket}/input/batch_requests_part",
                    file_name_suffix=".jsonl",
                    shard_name_template="-SSSSS-of-NNNNN"
                )
                
                results.bypassed | "Write Bypassed Predictions" >> beam.io.WriteToText(
                    f"gs://{clean_bucket}/output/bypassed/bypassed_predictions",
                    file_name_suffix=".jsonl",
                    shard_name_template="-SSSSS-of-NNNNN"
                )
                
            yield log_and_yield(self.name, "Inline Apache Beam pipeline finished execution!")
            
        except Exception as e:
            logger.error(f"Error executing Beam pipeline inline: {e}")
            yield log_and_yield(self.name, f"Pipeline Error: {e}")
            raise e

        # 6. Populate session state with generated GCS file URLs
        gcs_sources = []
        blobs = bucket.list_blobs(prefix="input/batch_requests_part")
        for b in blobs:
            if b.name.endswith(".jsonl"):
                gcs_sources.append(f"gs://{clean_bucket}/{b.name}")
                
        yield log_and_yield(self.name, f"Discovered {len(gcs_sources)} chunk files generated in GCS input folder.")
        
        ctx.session.state["gcs_sources"] = gcs_sources
        ctx.session.state["bucket_name"] = self.bucket_name
        ctx.session.state["bypassed_noise_saved"] = True
