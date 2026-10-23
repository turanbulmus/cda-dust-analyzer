import os
import json
import logging
from datetime import datetime
from typing import AsyncGenerator
from typing_extensions import override

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
import pandas as pd
from google.cloud import storage
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config
from ..tools.utils import parse_response, normalise_binary_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))


configs = Config()


def make_extract_clean_prediction_dofn():
    """Creates a local-scope DoFn with __module__ = '__main__' so cloudpickle serializes the class
    definition by value on Cloud Dataflow workers without requiring custom package imports."""

    class _ExtractCleanPredictionDoFn(beam.DoFn):
        def process(self, line):
            import json
            import re

            sclk_pat = re.compile(r"Sample ID \(sclk\): (\d+)")
            try:
                tail = line[-50000:] if len(line) > 50000 else line
                m_sclk = sclk_pat.search(tail)
                echoed_sclk = m_sclk.group(1) if m_sclk else None

                record = json.loads(line)
                if echoed_sclk is None:
                    req_parts = record.get("request", {}).get("contents", [{}])[-1].get("parts", [])
                    for p in reversed(req_parts):
                        m = sclk_pat.search(p.get("text", ""))
                        if m:
                            echoed_sclk = m.group(1)
                            break

                resp_parts = (
                    record.get("response", {})
                    .get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [])
                )
                resp_text = ""
                for p in reversed(resp_parts):
                    if "text" in p and not p.get("thought"):
                        resp_text = p["text"]
                        break

                clean_text = re.sub(r"```json\s*|\s*```", "", resp_text).strip()
                parsed = json.loads(clean_text) if clean_text else {}
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else {}
                if not isinstance(parsed, dict):
                    parsed = {}

                pred_id = echoed_sclk or parsed.get("id")
                raw_cls = (
                    str(parsed.get("class") or parsed.get("class_label") or "non-3-OH")
                    .strip()
                    .strip("'\"")
                )
                if raw_cls.lower().startswith("class "):
                    raw_cls = raw_cls[6:].strip()
                key = raw_cls.lower().replace(" ", "").replace("_", "-")
                pred_label = "3-OH" if key in ("3-oh", "3oh") else "non-3-OH"
                explanation = str(parsed.get("explanation", ""))

                if pred_id is not None:
                    clean_record = {
                        "response": {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            {
                                                "text": json.dumps(
                                                    {
                                                        "id": str(pred_id),
                                                        "class": pred_label,
                                                        "explanation": explanation,
                                                        "source": "vlm_argon",
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
                    yield json.dumps(clean_record)
            except Exception:
                pass

    _ExtractCleanPredictionDoFn.__module__ = "__main__"
    return _ExtractCleanPredictionDoFn()


class ResultAnalysisAgent(BaseAgent):
    """Extracts predictions in-cloud via Apache Beam (Cloud Dataflow), merges Stage-1 bypassed
    predictions, and computes binary 3-OH vs non-3-OH metrics plus 12-subclass breakdowns."""

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

        output_filepath = "cda_3oh_agent/data/output/predictions.jsonl"
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        if inference_path != "local":
            os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
            os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"

            bucket_name = ctx.session.state.get("bucket_name", configs.agent_settings.bucket_name)
            clean_bucket = bucket_name.replace("gs://", "").strip("/")
            output_prefix = configs.agent_settings.gcs_output_prefix.strip("/")

            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(clean_bucket)

            if os.path.exists(output_filepath):
                os.remove(output_filepath)

            existing_cleans = [
                b
                for b in bucket.list_blobs(prefix=f"{output_prefix}/clean_predictions")
                if b.name.endswith(".jsonl")
            ]
            for b in existing_cleans:
                b.delete()

            explicit_out_dir = ctx.session.state.get("gcs_output_directory")
            if explicit_out_dir:
                folder_prefix = explicit_out_dir.replace(f"gs://{clean_bucket}/", "").strip("/")
            else:
                blobs = list(bucket.list_blobs(prefix=output_prefix))
                prediction_blobs = [
                    b
                    for b in blobs
                    if b.name.endswith(".jsonl")
                    and "prediction" in b.name
                    and "clean_predictions" not in b.name
                    and "bypassed" not in b.name
                ]
                if not prediction_blobs:
                    yield log_and_yield(
                        self.name, "Warning: No raw prediction JSONL files found in GCS output directory."
                    )
                    return
                prediction_blobs.sort(key=lambda x: x.time_created, reverse=True)
                folder_prefix = os.path.dirname(prediction_blobs[0].name)

            test_mode = configs.agent_settings.test_mode
            runner = "DirectRunner" if test_mode else "DataflowRunner"
            job_name = f"cda-3oh-cleanup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            yield log_and_yield(
                self.name,
                f"Launching Apache Beam cleanup pipeline on {runner} (Job Name: {job_name}) "
                f"from gs://{clean_bucket}/{folder_prefix}/*.jsonl...",
            )

            options_args = [
                f"--runner={runner}",
                f"--project={self.project_id}",
                f"--temp_location=gs://{clean_bucket}/temp",
                f"--staging_location=gs://{clean_bucket}/staging",
            ]
            if runner == "DataflowRunner":
                options_args.extend(
                    [
                        f"--region={configs.agent_settings.location}",
                        f"--job_name={job_name}",
                        "--max_num_workers=10",
                        "--machine_type=e2-standard-4",
                    ]
                )

            pipeline_options = PipelineOptions(options_args)
            pipeline_options.view_as(SetupOptions).save_main_session = False

            input_wildcard = f"gs://{clean_bucket}/{folder_prefix}/*.jsonl"
            output_clean_prefix = f"gs://{clean_bucket}/{output_prefix}/clean_predictions/predictions"

            dofn = make_extract_clean_prediction_dofn()

            with beam.Pipeline(options=pipeline_options) as p:
                (
                    p
                    | "Read Raw Predictions" >> beam.io.ReadFromText(input_wildcard)
                    | "Extract Clean Metadata" >> beam.ParDo(dofn)
                    | "Write Clean Predictions"
                    >> beam.io.WriteToText(
                        output_clean_prefix,
                        file_name_suffix=".jsonl",
                        shard_name_template="-SSSSS-of-NNNNN",
                    )
                )

            yield log_and_yield(self.name, "Beam prediction extraction pipeline completed successfully!")

            clean_blobs = [
                b
                for b in bucket.list_blobs(prefix=f"{output_prefix}/clean_predictions/")
                if b.name.endswith(".jsonl")
            ]
            yield log_and_yield(
                self.name, f"Downloading {len(clean_blobs)} clean prediction shard(s) into {output_filepath}..."
            )
            with open(output_filepath, "wb") as outfile:
                for blob in sorted(clean_blobs, key=lambda x: x.name):
                    content = blob.download_as_bytes()
                    outfile.write(content)
                    if not content.endswith(b"\n"):
                        outfile.write(b"\n")

            bypassed_blobs = [
                b
                for b in bucket.list_blobs(prefix=f"{output_prefix}/bypassed/")
                if b.name.endswith(".jsonl")
            ]
            if bypassed_blobs:
                yield log_and_yield(
                    self.name,
                    f"Merging {len(bypassed_blobs)} Stage-1 bypassed shard(s) into {output_filepath}...",
                )
                with open(output_filepath, "ab") as outfile:
                    for blob in sorted(bypassed_blobs, key=lambda x: x.name):
                        content = blob.download_as_bytes()
                        outfile.write(content)
                        if not content.endswith(b"\n"):
                            outfile.write(b"\n")

        yield log_and_yield(self.name, f"Running inline binary & per-subclass evaluation on {output_filepath}...")
        try:
            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )
            import matplotlib.pyplot as plt
            import seaborn as sns

            preds = []
            with open(output_filepath, "r") as f:
                for line in f:
                    if line.strip():
                        preds.append(json.loads(line))

            df = pd.read_parquet("cda_3oh_agent/data/testing/cda_test.parquet")
            used_ids = set(str(u) for u in (ctx.session.state.get("used_ids") or []))
            df_eval = df[~df["sclk"].astype(str).isin(used_ids)].copy()

            truth_map = {}
            cat_map = {}
            for _, row in df_eval.iterrows():
                k = str(int(row["sclk"]))
                truth_map[k] = str(row["class"])
                cat_map[k] = str(row.get("catalogue_class", row["class"]))

            seen_ids = set()
            matched_rows = []
            for p in preds:
                try:
                    resp_text = p["response"]["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    resp_text = ""
                parsed = parse_response(resp_text)
                pred_id = parsed.get("id")
                if pred_id is None:
                    continue
                clean_id = str(int(float(pred_id))) if str(pred_id).replace(".", "").isdigit() else str(pred_id)
                if clean_id in seen_ids or clean_id not in truth_map:
                    continue
                seen_ids.add(clean_id)
                pred_label = normalise_binary_label(parsed.get("class"))
                expl = str(parsed.get("explanation", ""))
                source = "stage1_bypass" if "Stage-1 Pre-Filter" in expl else "vlm_argon"
                matched_rows.append(
                    {
                        "sclk": int(clean_id),
                        "catalogue_class": cat_map[clean_id],
                        "true_class": truth_map[clean_id],
                        "predicted_class": pred_label,
                        "source": source,
                        "explanation": expl,
                    }
                )

            results_df = pd.DataFrame(matched_rows)
            os.makedirs("cda_3oh_agent/data/results", exist_ok=True)
            results_df.to_csv("cda_3oh_agent/data/results/results.csv", index=False)
            results_df.to_parquet("cda_3oh_agent/data/results/results.parquet", index=False)

            y_true = results_df["true_class"].tolist()
            y_pred = results_df["predicted_class"].tolist()
            target_names = ["3-OH", "non-3-OH"]

            report = classification_report(y_true, y_pred, labels=target_names, digits=4)
            cm = confusion_matrix(y_true, y_pred, labels=target_names)
            acc = accuracy_score(y_true, y_pred)
            prec_3oh = precision_score(y_true, y_pred, pos_label="3-OH", zero_division=0)
            rec_3oh = recall_score(y_true, y_pred, pos_label="3-OH", zero_division=0)
            f1_3oh = f1_score(y_true, y_pred, pos_label="3-OH", zero_division=0)
            macro_f1 = f1_score(y_true, y_pred, labels=target_names, average="macro", zero_division=0)

            sub_rows = []
            for cat_cls, grp in results_df.groupby("catalogue_class"):
                n_tot = len(grp)
                n_pred_3oh = int((grp["predicted_class"] == "3-OH").sum())
                n_pred_non = int((grp["predicted_class"] == "non-3-OH").sum())
                n_bypassed = int((grp["source"] == "stage1_bypass").sum())
                true_bin = grp["true_class"].iloc[0]
                correct = n_pred_3oh if true_bin == "3-OH" else n_pred_non
                sub_rows.append(
                    {
                        "catalogue_class": cat_cls,
                        "true_binary": true_bin,
                        "total": n_tot,
                        "stage1_bypassed": n_bypassed,
                        "vlm_evaluated": n_tot - n_bypassed,
                        "pred_3-OH": n_pred_3oh,
                        "pred_non-3-OH": n_pred_non,
                        "accuracy": round(correct / n_tot, 4) if n_tot else 0.0,
                    }
                )
            sub_df = pd.DataFrame(sub_rows).sort_values(["true_binary", "total"], ascending=[True, False])
            sub_df.to_csv("cda_3oh_agent/data/results/per_class_breakdown.csv", index=False)

            metrics_summary = {
                "model": configs.agent_settings.model,
                "total_evaluated": len(results_df),
                "stage1_bypassed": int((results_df["source"] == "stage1_bypass").sum()),
                "vlm_evaluated": int((results_df["source"] == "vlm_argon").sum()),
                "accuracy": round(float(acc), 4),
                "precision_3oh": round(float(prec_3oh), 4),
                "recall_3oh": round(float(rec_3oh), 4),
                "f1_3oh": round(float(f1_3oh), 4),
                "macro_f1": round(float(macro_f1), 4),
                "tp_3oh": int(cm[0, 0]),
                "fn_3oh": int(cm[0, 1]),
                "fp_3oh": int(cm[1, 0]),
                "tn_3oh": int(cm[1, 1]),
            }
            with open("cda_3oh_agent/data/results/metrics_3oh_agent.json", "w") as f:
                json.dump(metrics_summary, f, indent=2)

            plt.figure(figsize=(7, 5.5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names,
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"CDA 3-OH Agent ({configs.agent_settings.model}) Confusion Matrix")
            plt.tight_layout()
            plt.savefig("cda_3oh_agent/data/results/confusion_matrix.png", dpi=150)
            plt.close()

            analysis_text = (
                f"\n--- Binary 3-OH vs non-3-OH Evaluation Results ({len(results_df)} spectra) ---\n"
                f"{report}\n"
                f"Confusion Matrix (Rows=True [3-OH, non-3-OH], Cols=Pred [3-OH, non-3-OH]):\n{cm}\n\n"
                f"Per-Catalogue-Subclass Breakdown:\n{sub_df.to_string(index=False)}\n"
            )
            yield log_and_yield(self.name, analysis_text)

        except Exception as e:
            yield log_and_yield(self.name, f"Error running analysis: {e}")
            raise e
