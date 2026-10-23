import os
import json
import asyncio
import logging
from typing import AsyncGenerator
from typing_extensions import override

import numpy as np
import pandas as pd
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import Client
from google.genai.types import Content, Part, GenerateContentConfig, ThinkingConfig

from ..config import Config
from ..prompts import (
    STAGE2_3OH_ADJUDICATOR_SYSTEM_PROMPT,
    STAGE2_3OH_ADJUDICATOR_TASK_PROMPT,
    CATALOGUE_SUBCLASS_DESCRIPTIONS,
)
from ..tools.utils import (
    extract_mass_telemetry_3oh,
    format_telemetry_table_3oh,
    generate_dual_panel_mass_spectrum_bytes,
    normalise_binary_label,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))


configs = Config()

STAGE2_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "id": {"type": "STRING", "description": "The provided sclk id"},
        "class": {
            "type": "STRING",
            "enum": ["3-OH", "non-3-OH"],
            "description": "Stage-2 specialist adjudicated binary class label",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence in the classification from 0.0 to 1.0",
        },
        "explanation": {
            "type": "STRING",
            "description": "2-3 sentence specialist explanation citing calibrated masses (u) and telemetry values",
        },
    },
    "required": ["id", "class", "confidence", "explanation"],
}


def build_stage2_hard_negative_exemplars(train_df: pd.DataFrame) -> list:
    """Builds the multimodal few-shot parts for Stage-2 intra-Type-3 specialist adjudication
    using Dual-Panel Mass-Calibrated Plots and 1D Quantitative Telemetry from the training set."""
    # Reduce to 8 highest-contrast exemplars (3 3-OH, 2 3-Car, 1 3-KNa, 1 3-Cl, 1 Type 1) for fast, focused inference
    selected_rows = []
    for cls_name, cap in [("3-OH", 3), ("3-Car", 2), ("3-KNa", 1), ("3-Cl", 1), ("1", 1)]:
        sub_c = train_df[train_df["catalogue_class"] == cls_name]
        if len(sub_c) > 0:
            selected_rows.append(sub_c.iloc[:cap])
    sub_df = pd.concat(selected_rows, ignore_index=True)

    parts = [
        Part.from_text(text=STAGE2_3OH_ADJUDICATOR_TASK_PROMPT),
        Part.from_text(
            text="\n=== STAGE-2 MASS-CALIBRATED REFERENCE EXEMPLARS (3-OH vs Hard-Negative Look-Alikes) ==="
        ),
    ]

    for idx, (_, row) in enumerate(sub_df.iterrows()):
        sclk_id = str(int(row["sclk"]))
        bin_lbl = str(row["class"])
        cat_cls = str(row["catalogue_class"])
        cat_desc = CATALOGUE_SUBCLASS_DESCRIPTIONS.get(cat_cls, cat_cls)
        spec = np.asarray(row["spectrum"], dtype=float)

        tele = extract_mass_telemetry_3oh(spec)
        tele_text = format_telemetry_table_3oh(tele)
        img_bytes = generate_dual_panel_mass_spectrum_bytes(
            spec, title=f"Reference {idx + 1} (SCLK {sclk_id})", mode="3oh"
        )

        if bin_lbl == "3-OH":
            rationale = (
                f"True 3-OH exemplar: strong Na+ (23u) and Na+(NaOH) (63u, prom={tele['prom_63']:.3f}), "
                f"clean hydroxide decay (103u prom={tele['prom_103']:.3f}), and flat 129u carbonate region "
                f"(excess_129={tele['excess_129']:.4f} <= 0.037)."
            )
        elif cat_cls == "3-Car":
            rationale = (
                f"Hard-Negative 3-Car (Carbonate) look-alike: shares Na+ (23u) and 63u/103u hydroxides, "
                f"but exhibits a distinct Na+(Na2CO3) peak in the red 129u band (excess_129={tele['excess_129']:.4f}, "
                f"prom_129={tele['prom_129']:.3f} @ {tele['pk_118_135_u']:.1f}u) and/or 85u Na+(Na2O) excess ({tele['excess_85']:.3f}). Must be rejected as non-3-OH."
            )
        elif cat_cls in ("3-KNa", "3-K"):
            rationale = (
                f"Hard-Negative {cat_cls} (Potassium-bearing salt): elevated K+ (39u) and potassium clusters at "
                f"95u (prom={tele['prom_95']:.3f}) / 113u (prom={tele['prom_113']:.3f}). Must be rejected as non-3-OH."
            )
        elif cat_cls == "3-Cl":
            rationale = (
                f"Hard-Negative 3-Cl (Chloride): strong Na+(NaCl) at 81u (prom={tele['prom_81']:.3f}) "
                f"and/or Na+(NaCl)2 at 139u (prom={tele['prom_139']:.3f}). Must be rejected as non-3-OH."
            )
        else:
            rationale = (
                f"Hard-Negative Type {cat_cls} (Water-ice / organic): strong H3O+ at 19u (h={tele['h19']:.2f}) "
                f"and water-cluster series at 55u (prom={tele['prom_55']:.3f}) and 73u (prom={tele['prom_73']:.3f}). Must be rejected as non-3-OH."
            )

        header = (
            f"\nReference Exemplar {idx + 1} (SCLK {sclk_id}) -> Ground Truth: {bin_lbl} "
            f"(Catalogue Subclass: '{cat_cls}' — {cat_desc})\n"
            f"{tele_text}\n"
            f"- Specialist Adjudication Rationale: {rationale}"
        )
        parts.append(Part.from_text(text=header))
        parts.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

    parts.append(
        Part.from_text(
            text="\n=== TARGET CANDIDATE FOR STAGE-2 SPECIALIST ADJUDICATION ==="
        )
    )
    return parts


async def adjudicate_candidates_3oh(
    candidates_df: pd.DataFrame,
    train_df: pd.DataFrame,
    project_id: str,
    model_id: str = "gemini-4-argon",
    concurrency: int = 8,
) -> pd.DataFrame:
    """Runs Stage-2 Specialist VLM Adjudication + 1D Physics Veto on a DataFrame of candidate spectra."""
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["GOOGLE_API_USE_MTLS"] = "never"
    os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

    client = Client(vertexai=True, project=project_id, location="global")
    exemplar_parts = build_stage2_hard_negative_exemplars(train_df)
    sem = asyncio.Semaphore(concurrency)

    async def process_one(row):
        async with sem:
            sclk_id = str(int(row["sclk"]))
            spec = np.asarray(row["spectrum"], dtype=float)
            raw_spec = row.get("raw_spectrum", None)

            tele = extract_mass_telemetry_3oh(spec, raw_spectrum=raw_spec)
            tele_text = format_telemetry_table_3oh(tele)
            img_bytes = generate_dual_panel_mass_spectrum_bytes(
                spec, title=f"Candidate SCLK {sclk_id}", mode="3oh"
            )

            target_parts = list(exemplar_parts) + [
                Part.from_text(
                    text=f"Candidate Sample ID (sclk): {sclk_id}\n{tele_text}\n"
                    f"Stage-1 Explanation: {row.get('explanation', '')}"
                ),
                Part.from_bytes(data=img_bytes, mime_type="image/png"),
            ]

            vlm_cls = "non-3-OH"
            vlm_conf = 0.5
            vlm_expl = ""
            for attempt in range(3):
                try:
                    resp = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=model_id,
                            contents=target_parts,
                            config=GenerateContentConfig(
                                system_instruction=STAGE2_3OH_ADJUDICATOR_SYSTEM_PROMPT,
                                response_mime_type="application/json",
                                response_schema=STAGE2_RESPONSE_SCHEMA,
                                temperature=0.0,
                            ),
                        ),
                        timeout=60.0,
                    )
                    parsed = json.loads(resp.text.strip())
                    if isinstance(parsed, list):
                        parsed = parsed[0] if parsed else {}
                    vlm_cls = normalise_binary_label(parsed.get("class"))
                    vlm_conf = float(parsed.get("confidence", 0.5))
                    vlm_expl = str(parsed.get("explanation", ""))
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"Stage-2 VLM error for SCLK {sclk_id}: {e}")
                        vlm_cls = "3-OH" if tele["passes_129_veto"] else "non-3-OH"
                        vlm_expl = f"Fallback to 1D physics gate due to API timeout/error: {e}"
                    else:
                        await asyncio.sleep(2 * (attempt + 1))

            physics_cls = "3-OH" if tele["passes_129_veto"] else "non-3-OH"
            strict_cls = "3-OH" if tele["passes_strict_veto"] else "non-3-OH"
            # Hybrid Neuro-Symbolic Consensus: must pass both the 129u Carbonate Physics Gate and Stage-2 Specialist VLM
            hybrid_cls = "3-OH" if (vlm_cls == "3-OH" and tele["passes_129_veto"]) else "non-3-OH"
            hybrid_strict_cls = "3-OH" if (vlm_cls == "3-OH" and tele["passes_strict_veto"]) else "non-3-OH"

            out = {
                "sclk": int(row["sclk"]),
                "stage2_vlm_class": vlm_cls,
                "stage2_vlm_confidence": round(vlm_conf, 3),
                "stage2_vlm_explanation": vlm_expl,
                "physics_veto_class": physics_cls,
                "strict_veto_class": strict_cls,
                "hybrid_class": hybrid_cls,
                "hybrid_strict_class": hybrid_strict_cls,
            }
            out.update(tele)
            return out

    tasks = [process_one(row) for _, row in candidates_df.iterrows()]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


class Stage2SpecialistAdjudicatorAgent(BaseAgent):
    """Runs Stage-2 Intra-Type-3 Specialist Adjudication (Dual-Panel Mass-Calibrated Plots +
    1D Quantitative Telemetry + Physics Veto Gate) on all candidates flagged by Stage 1."""

    project_id: str
    model_id: str = "gemini-4-argon"

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if not configs.agent_settings.enable_stage2_adjudicator:
            yield log_and_yield(self.name, "Stage-2 Specialist Adjudicator is disabled; skipping.")
            return

        results_csv = "cda_3oh_agent/data/results/results.csv"
        if not os.path.exists(results_csv):
            yield log_and_yield(self.name, f"No Stage-1 results found at {results_csv}; skipping Stage-2.")
            return

        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        res_df = pd.read_csv(results_csv)
        test_df = pd.read_parquet("cda_3oh_agent/data/testing/cda_test.parquet")
        train_df = pd.read_parquet("cda_3oh_agent/data/raw/cda_train.parquet")

        flagged_sclks = set(res_df[res_df["predicted_class"] == "3-OH"]["sclk"].astype(int).tolist())
        yield log_and_yield(
            self.name,
            f"Running Stage-2 Specialist Adjudicator ({self.model_id}) + Two-Anchor Mass Calibration "
            f"on {len(flagged_sclks)} Stage-1 positive candidates...",
        )

        cand_df = res_df[res_df["sclk"].isin(flagged_sclks)].merge(
            test_df[["sclk", "spectrum", "qi_ampl"]], on="sclk", how="left"
        )

        adj_df = await adjudicate_candidates_3oh(
            candidates_df=cand_df,
            train_df=train_df,
            project_id=self.project_id,
            model_id=self.model_id,
        )

        merged_cand = cand_df.drop(columns=["spectrum"]).merge(adj_df, on="sclk", how="left")
        os.makedirs("cda_3oh_agent/data/results", exist_ok=True)
        merged_cand.to_csv("cda_3oh_agent/data/results/stage2_adjudication_results.csv", index=False)

        # Merge back into the full 18,305 archive evaluation
        sclk_to_vlm = dict(zip(adj_df["sclk"], adj_df["stage2_vlm_class"]))
        sclk_to_phys = dict(zip(adj_df["sclk"], adj_df["physics_veto_class"]))
        sclk_to_hyb = dict(zip(adj_df["sclk"], adj_df["hybrid_class"]))

        res_df["pred_stage1"] = res_df["predicted_class"]
        res_df["pred_physics_veto"] = res_df.apply(
            lambda r: sclk_to_phys.get(int(r["sclk"]), r["predicted_class"]), axis=1
        )
        res_df["pred_stage2_vlm"] = res_df.apply(
            lambda r: sclk_to_vlm.get(int(r["sclk"]), r["predicted_class"]), axis=1
        )
        res_df["pred_hybrid"] = res_df.apply(
            lambda r: sclk_to_hyb.get(int(r["sclk"]), r["predicted_class"]), axis=1
        )
        res_df.to_csv("cda_3oh_agent/data/results/results_hybrid_stage2.csv", index=False)

        y_true = res_df["true_class"].tolist()
        comparison = {}
        for col, label in [
            ("pred_stage1", "Stage-1 VLM Only (Gemini 4 Argon + Referee Prompt)"),
            ("pred_physics_veto", "Stage-1 VLM + Two-Anchor 129u Physics Veto"),
            ("pred_stage2_vlm", "Stage-1 VLM + Stage-2 Specialist VLM Adjudicator"),
            ("pred_hybrid", "Stage-1 VLM + Hybrid Neuro-Symbolic Stage-2 (VLM + Physics Gate)"),
        ]:
            y_p = res_df[col].tolist()
            tp = int(((res_df["true_class"] == "3-OH") & (res_df[col] == "3-OH")).sum())
            fn = int(((res_df["true_class"] == "3-OH") & (res_df[col] == "non-3-OH")).sum())
            fp = int(((res_df["true_class"] == "non-3-OH") & (res_df[col] == "3-OH")).sum())
            tn = int(((res_df["true_class"] == "non-3-OH") & (res_df[col] == "non-3-OH")).sum())
            fp_by_cls = (
                res_df[(res_df["true_class"] == "non-3-OH") & (res_df[col] == "3-OH")]["catalogue_class"]
                .value_counts()
                .to_dict()
            )
            comparison[col] = {
                "description": label,
                "accuracy": round(float(accuracy_score(y_true, y_p)), 4),
                "precision_3oh": round(float(precision_score(y_true, y_p, pos_label="3-OH", zero_division=0)), 4),
                "recall_3oh": round(float(recall_score(y_true, y_p, pos_label="3-OH", zero_division=0)), 4),
                "f1_3oh": round(float(f1_score(y_true, y_p, pos_label="3-OH", zero_division=0)), 4),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "fp_by_catalogue_class": {str(k): int(v) for k, v in fp_by_cls.items()},
            }

        with open("cda_3oh_agent/data/results/metrics_3oh_stage2_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        yield log_and_yield(
            self.name,
            f"Stage-2 Specialist Adjudication Complete!\n{json.dumps(comparison, indent=2)}",
        )
