import os
import argparse
import logging
import json

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            # Bypassed noise: Format immediately to VLM output schema
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


class CdaDataPrepOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument("--input_parquet", required=True, help="GCS or local path to input parquet file.")
        parser.add_argument("--few_shot_gcs", required=True, help="GCS path to cached_examples.jsonl.")
        parser.add_argument("--output_queued_prefix", required=True, help="GCS output prefix for VLM request files.")
        parser.add_argument("--output_bypassed_prefix", required=True, help="GCS output prefix for bypassed predictions.")
        parser.add_argument("--limit", type=int, default=0, help="Optional limit for testing.")
        parser.add_argument("--project_id", default="turan-genai-bb", help="GCP project ID.")

def run(argv=None):
    pipeline_options = PipelineOptions(argv)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    
    cda_options = pipeline_options.view_as(CdaDataPrepOptions)
    
    # Prompt imports to pass to workers
    from cda_dust_agent.prompts import SYSTEM_INSTRUCTION_TEXT, CLASSIFICATION_USER_PROMPT, GENERAL_PROFILES
    
    # Read Parquet at driver (since 18K rows is tiny)
    import pandas as pd
    logger.info(f"Driver loading input parquet from {cda_options.input_parquet}...")
    df = pd.read_parquet(cda_options.input_parquet)
    
    if cda_options.limit > 0 and cda_options.limit < len(df):
        logger.info(f"Driver sampling {cda_options.limit} records...")
        df = df.sample(n=cda_options.limit, random_state=42)
        
    # Convert numpy types to native Python types to avoid serialization/version issues across worker nodes
    records = []
    for _, row in df.iterrows():
        records.append({
            "sclk": int(row["sclk"]),
            "spectrum": [float(v) for v in row["spectrum"]],
            "qi_ampl": float(row["qi_ampl"]) if pd.notna(row.get("qi_ampl", 0.0)) else 0.0,
            "class": str(row.get("class", "Noise"))
        })
    logger.info(f"Driver converted {len(records)} records with native Python types for distributed processing.")
    
    with beam.Pipeline(options=pipeline_options) as p:
        # Create PCollection from records
        dist_records = p | "Create Records" >> beam.Create(records)
        
        # Apply parallel pre-filtering & plotting DoFn
        results = dist_records | "Filter and Plot" >> beam.ParDo(
            FilterAndPlotDoFn(
                few_shot_gcs_path=cda_options.few_shot_gcs,
                system_instruction=SYSTEM_INSTRUCTION_TEXT,
                user_prompt=CLASSIFICATION_USER_PROMPT,
                general_profiles=GENERAL_PROFILES,
                project_id=cda_options.project_id
            )
        ).with_outputs("queued", "bypassed")
        
        # Write VLM batch requests to GCS
        (
            results.queued 
            | "Write Queued Requests" >> beam.io.WriteToText(
                cda_options.output_queued_prefix,
                file_name_suffix=".jsonl",
                shard_name_template="-SSSSS-of-NNNNN"
            )
        )
        
        # Write bypassed predictions to GCS
        (
            results.bypassed
            | "Write Bypassed Predictions" >> beam.io.WriteToText(
                cda_options.output_bypassed_prefix,
                file_name_suffix=".jsonl",
                shard_name_template="-SSSSS-of-NNNNN"
            )
        )
        
    logger.info("Dataflow Data Prep pipeline completed successfully!")

if __name__ == "__main__":
    run()
