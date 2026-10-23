import os
import json
import logging
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
import pandas as pd
import numpy as np
from google.cloud import storage
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
from ..prompts import (
    CLASSIFICATION_TASK_PROMPT,
    get_active_system_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))


configs = Config()


def make_generate_batch_request_dofn(base_parts, system_prompt, num_examples):
    """Creates a local-scope DoFn with __module__ = '__main__' so cloudpickle serializes the class
    definition by value on Cloud Dataflow workers without requiring custom package imports."""

    class _GenerateBatchRequestDoFn(beam.DoFn):
        def __init__(self, bp, sp, ne):
            self.base_parts = bp
            self.system_prompt = sp
            self.num_examples = ne

        def process(self, row):
            import io
            import json
            import base64
            import numpy as np
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.ticker import MultipleLocator

            sclk_id = str(int(row["sclk"]))
            spectrum = np.array(row["spectrum"], dtype=float)

            # Stage-1 Pre-Filter (calibrated so 0% of true 3-OH spectra are bypassed)
            is_noise = False
            if len(spectrum) == 0:
                is_noise = True
            else:
                std_val = float(np.std(spectrum))
                max_val = float(np.max(spectrum))
                mean_val = float(np.mean(spectrum))
                q99_val = float(np.percentile(spectrum, 99))
                if std_val < 0.08:
                    is_noise = True
                elif (max_val - mean_val) < 0.25:
                    is_noise = True
                elif (max_val - q99_val) < 0.05 and std_val < 0.22 and (max_val - mean_val) < 0.40:
                    is_noise = True

            if is_noise:
                bypassed_record = {
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": json.dumps(
                                                {
                                                    "id": sclk_id,
                                                    "class": "non-3-OH",
                                                    "explanation": (
                                                        "Stage-1 Pre-Filter: Bypassed VLM. Flatline/Low-variance "
                                                        "noise spectrum automatically classified as non-3-OH."
                                                    ),
                                                    "k": self.num_examples,
                                                }
                                            )
                                        }
                                    ],
                                    "role": "model",
                                }
                            }
                        ]
                    }
                }
                yield beam.pvalue.TaggedOutput("bypassed", json.dumps(bypassed_record))
                return

            # Render 0.0-6.4 us Time-of-Flight spectrum (720x360 px at 60 DPI)
            fig = Figure(figsize=(12, 6), dpi=60)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            t = np.arange(10, 10 + len(spectrum)) * 0.01  # 100 MHz -> 0.01 us per sample
            ax.plot(t, spectrum, color="black", linewidth=1.5)
            ax.set_title(f"Sample {sclk_id}", fontsize=14)
            ax.set_xlim(0.0, 6.4)
            ax.set_ylim(-0.02, 1.02)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.grid(True, which="major", linewidth=0.8)
            ax.grid(True, which="minor", linewidth=0.3, alpha=0.5)
            ax.set_xlabel("Time of flight (µs)", fontsize=13)
            ax.set_ylabel("log amplitude (scaled)", fontsize=13)
            ax.tick_params(labelsize=12)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            parts = list(self.base_parts)
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64_img}})
            parts.append({"text": f"Sample ID (sclk): {sclk_id} | k: {self.num_examples}"})

            request = {
                "request": {
                    "contents": [{"role": "user", "parts": parts}],
                    "systemInstruction": {"parts": [{"text": self.system_prompt}]},
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "STRING", "description": "The provided sclk id"},
                                "class": {
                                    "type": "STRING",
                                    "enum": ["3-OH", "non-3-OH"],
                                    "description": "The predicted binary label",
                                },
                                "explanation": {
                                    "type": "STRING",
                                    "description": "Brief explanation based on peak times in us",
                                },
                                "k": {
                                    "type": "INTEGER",
                                    "description": "The number of reference examples shown",
                                },
                            },
                            "required": ["id", "class", "explanation", "k"],
                        },
                    },
                }
            }
            yield beam.pvalue.TaggedOutput("queued", json.dumps(request))

    _GenerateBatchRequestDoFn.__module__ = "__main__"
    return _GenerateBatchRequestDoFn(base_parts, system_prompt, num_examples)


class DataPrepAgent(BaseAgent):
    """Formats the test dataset into Vertex AI Batch Prediction JSONL shards via Cloud Dataflow
    and applies the Stage-1 `is_obvious_noise_3oh()` pre-filter."""

    project_id: str
    bucket_name: str

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        inference_path = configs.agent_settings.inference_path
        if inference_path not in ["local", "batch"]:
            return

        if ctx.session.state.get("fsa_state") != "done":
            yield log_and_yield(self.name, "Few-shot annotation did not complete; skipping data prep.")
            return

        examples = ctx.session.state.get("examples", [])
        used_ids = set(ctx.session.state.get("used_ids", []))
        system_prompt = ctx.session.state.get("optimized_system_prompt") or get_active_system_prompt()

        df = pd.read_parquet("cda_3oh_agent/data/testing/cda_test.parquet")
        test_df = df[~df["sclk"].astype(str).isin(used_ids)].copy()

        test_n = configs.agent_settings.test_n
        if test_n > 0:
            test_df = test_df.sample(n=min(test_n, len(test_df)), random_state=42)

        yield log_and_yield(
            self.name,
            f"Preparing {len(test_df)} test samples with {len(examples)} few-shot exemplars "
            f"(Stage-1 pre-filter + Cloud Dataflow enabled)...",
        )

        base_parts = [{"text": CLASSIFICATION_TASK_PROMPT}, {"text": "Here are reference examples:"}]
        for ex in examples:
            base_parts.append({"text": ex.get("formatted_text", ex["explanation"])})
            base_parts.append({"inline_data": {"mime_type": "image/png", "data": ex["image_b64"]}})
        base_parts.append({"text": "Now, analyze the following spectrum:"})

        rows_to_process = []
        for _, row in test_df.iterrows():
            rows_to_process.append(
                {
                    "sclk": int(row["sclk"]),
                    "spectrum": np.asarray(row["spectrum"], dtype=float).tolist(),
                }
            )

        clean_bucket = self.bucket_name.replace("gs://", "").strip("/")
        input_prefix = configs.agent_settings.gcs_input_prefix.strip("/")
        output_prefix = configs.agent_settings.gcs_output_prefix.strip("/")

        os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
        os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"

        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(clean_bucket)

        old_inputs = list(bucket.list_blobs(prefix=f"{input_prefix}/batch_requests"))
        for b in old_inputs:
            b.delete()
        old_bypassed = list(bucket.list_blobs(prefix=f"{output_prefix}/bypassed/"))
        for b in old_bypassed:
            b.delete()

        output_gcs_prefix = f"gs://{clean_bucket}/{input_prefix}/batch_requests"
        bypassed_gcs_prefix = f"gs://{clean_bucket}/{output_prefix}/bypassed/bypassed_predictions"

        test_mode = configs.agent_settings.test_mode
        runner = "DirectRunner" if test_mode else "DataflowRunner"
        job_name = f"cda-3oh-prep-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        options_args = [
            f"--runner={runner}",
            f"--project={self.project_id}",
            f"--temp_location=gs://{clean_bucket}/temp",
            f"--staging_location=gs://{clean_bucket}/staging",
        ]
        if runner == "DataflowRunner":
            req_file = "/tmp/cda_dataflow_requirements.txt"
            with open(req_file, "w") as rf:
                rf.write("matplotlib\n")
            options_args.extend(
                [
                    f"--region={configs.agent_settings.location}",
                    f"--job_name={job_name}",
                    f"--requirements_file={req_file}",
                    "--max_num_workers=10",
                    "--machine_type=e2-standard-4",
                ]
            )

        pipeline_options = PipelineOptions(options_args)
        pipeline_options.view_as(SetupOptions).save_main_session = False

        yield log_and_yield(
            self.name,
            f"Launching Apache Beam pipeline on {runner} (job_name={job_name}) -> {output_gcs_prefix}...",
        )

        dofn = make_generate_batch_request_dofn(base_parts, system_prompt, len(examples))

        with beam.Pipeline(options=pipeline_options) as p:
            results = (
                p
                | "Create Rows" >> beam.Create(rows_to_process)
                | "Reshuffle Rows" >> beam.Reshuffle()
                | "Render & Stage-1 Filter" >> beam.ParDo(dofn).with_outputs("queued", "bypassed")
            )
            _ = results.queued | "Write Queued Shards" >> beam.io.WriteToText(
                output_gcs_prefix,
                file_name_suffix=".jsonl",
                shard_name_template="_part-SSSSS-of-NNNNN",
                num_shards=20 if not test_mode else 2,
            )
            _ = results.bypassed | "Write Bypassed Shards" >> beam.io.WriteToText(
                bypassed_gcs_prefix,
                file_name_suffix=".jsonl",
                shard_name_template="_part-SSSSS-of-NNNNN",
                num_shards=5 if not test_mode else 1,
            )

        input_blobs = [
            f"gs://{clean_bucket}/{b.name}"
            for b in bucket.list_blobs(prefix=f"{input_prefix}/batch_requests")
            if b.name.endswith(".jsonl")
        ]
        bypassed_blobs = [
            f"gs://{clean_bucket}/{b.name}"
            for b in bucket.list_blobs(prefix=f"{output_prefix}/bypassed/")
            if b.name.endswith(".jsonl")
        ]
        yield log_and_yield(
            self.name,
            f"Dataflow pipeline finished! Generated {len(input_blobs)} queued JSONL shard(s) "
            f"and {len(bypassed_blobs)} Stage-1 bypassed shard(s).",
        )
        ctx.session.state["gcs_source_uris"] = sorted(input_blobs)
        ctx.session.state["bucket_name"] = self.bucket_name
        ctx.session.state["dp_state"] = "done"
