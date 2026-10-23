# Cassini CDA Binary `3-OH` Classification Study (`gemini-3.5-flash` vs. `gemini-4-argon` & `cda_3oh_agent`)

This directory contains the forked Google ADK multi-agent pipeline (`cda_3oh_agent`) for binary **`3-OH` (sodium-hydroxide-dominated salt-rich Type 3 grains) vs. `non-3-OH`** discovery in the Cassini Cosmic Dust Analyzer (CDA) archive, along with the full evaluation results across **`gemini-3.5-flash`** and **`gemini-4-argon`**.

---

## 1. Executive Summary of Study Findings

1. **Model Upgrade (`gemini-3.5-flash` $\rightarrow$ `gemini-4-argon`)**:
   - On the held-out `698`–`707` labelled study benchmark, switching from `gemini-3.5-flash` to `gemini-4-argon` improves binary accuracy and cuts false positives across both prompt types:
     - **Human-in-the-Loop (HITL) Prompt (`707` study spectra)**: Accuracy rises from **`83.95%` $\rightarrow$ `87.41%`**, `3-OH` Recall from **`90.24%` $\rightarrow$ `92.68%` (`38/41`)**, and false positives drop from `108` $\rightarrow$ `84`.
     - **AI-Only Literature Prompt (`698` study spectra)**: Accuracy rises from **`93.84%` $\rightarrow$ `95.99%`**, `3-OH` Precision jumps from **`48.75%` $\rightarrow$ `60.32%`**, `3-OH` F1 improves from **`0.6446` $\rightarrow$ `0.7308`**, and false positives drop by **39%** (`41` $\rightarrow$ `25`).
2. **Anchored Multi-Agent Prompt Optimization (`cda_3oh_agent` + `gemini-4-argon`)**:
   - Using the AI-generated literature prompt (`ANCHOR_3OH_AI_PROMPT`) as an anchoring point, the 3-agent loop (`AcademicResearchAgent` $\rightarrow$ `AcademicCriticAgent` $\rightarrow$ `AcademicRefereeAgent`) on `gemini-4-argon` synthesized a 5-gate diagnostic decision tree ([`refined_prompt.txt`](refined_prompt.txt)).
   - On the same **`698`-sample study benchmark**, `cda_3oh_agent` achieved the best overall performance:
     - **Accuracy**: **`96.42%`** (`673 / 698`)
     - **`3-OH` Precision**: **`63.79%`** (`37` TP, `21` FP)
     - **`3-OH` Recall**: **`90.24%`** (`37 / 41`)
     - **`3-OH` F1**: **`0.7475`** | **Macro F1**: **`0.8641`**
     - Eliminated **100% of false positives** on `Type 1` (water ice), `Type 2` (organic ice), `3-Cl` (sodium chloride), `3-K` (potassium salt), `3-P` (phosphate), `4`, `5`, `5-Na`, and `Noise`, and cut `3-KNa` false positives from `6` $\rightarrow$ `1`.
3. **Full Labelled Archive Evaluation (`18,305` Unique Labelled Test Spectra)**:
   - Running `cda_3oh_agent` end-to-end across all `18,305` held-out labelled test spectra (`41` `3-OH`, `18,264` `non-3-OH` — a `1:445` imbalance):
     - **Overall Accuracy**: **`99.83%`** (`18,274 / 18,305`)
     - **`3-OH` Recall**: **`90.24%`** (`37 / 41`)
     - **`3-OH` Precision**: **`57.81%`** (`37` TP, only **`27` FP out of `18,264` non-3-OH spectra** — a **`0.15%` false-positive rate**)
     - **`3-OH` F1**: **`0.7048`** | **Macro F1**: **`0.8520`**
     - **Stage-1 Pre-Filter (`is_obvious_noise`)**: Bypassed **`4,202` spectra (`22.95%`)** as obvious noise with **`0` true `3-OH` spectra lost**, saving ~23% of VLM inference cost.
4. **Unlabelled $Q_I$-Filtered Archive (`24,829` Spectra in `Notebooks/3oh_classification/`)**:
   - Across all `24,829` unlabelled $Q_I$-filtered spectra (`batches 00..24`), **61 high-confidence `3-OH` candidates** were agreed upon by all three models/runs (`gemini-3.5-flash` HITL, `gemini-3.5-flash` AI-Only, and `gemini-4-argon` HITL); see [`Notebooks/3oh_classification/inference_results_3oh_argon/candidates_3oh_comparison.csv`](../Notebooks/3oh_classification/inference_results_3oh_argon/candidates_3oh_comparison.csv).

---

## 2. Benchmark Comparison on Held-Out Study Set (`698`–`707` Labelled Spectra)

| Configuration | Model | Study $N$ | Accuracy | `3-OH` Recall | `3-OH` Precision | `3-OH` F1 | Macro F1 | False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HITL Prompt (Nozair 9-shot + 9 neg) | `gemini-3.5-flash` | 707 | `83.95%` | `90.24%` (37/41) | `25.52%` | `0.3978` | `0.6531` | 108 |
| HITL Prompt (Nozair 9-shot + 9 neg) | `gemini-4-argon` | 707 | `87.41%` | `92.68%` (38/41) | `31.15%` | `0.4663` | `0.6976` | 84 |
| AI-Only Anchor Prompt (18-shot) | `gemini-3.5-flash` | 698 | `93.84%` | **`95.12%` (39/41)** | `48.75%` | `0.6446` | `0.8065` | 41 |
| AI-Only Anchor Prompt (18-shot) | `gemini-4-argon` | 698 | `95.99%` | `92.68%` (38/41) | `60.32%` | `0.7308` | `0.8545` | 25 |
| **`cda_3oh_agent` (Referee-Optimized)** | **`gemini-4-argon`** | **698** | **`96.42%`** | **`90.24%` (37/41)** | **`63.79%`** | **`0.7475`** | **`0.8641`** | **21** |

### Per-Subclass False-Positive Breakdown on the `698` Study Benchmark

| Catalogue Subclass | Study $N$ | `3.5-flash` (HITL) | `4-argon` (HITL) | `3.5-flash` (AI-Only) | `4-argon` (AI-Only) | **`cda_3oh_agent` (`4-argon`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`3-OH` (True Positives / 41)** | **41** | **37 (90.2%)** | **38 (92.7%)** | **39 (95.1%)** | **38 (92.7%)** | **37 (90.2%)** |
| `3-Car` (False Positives) | 85 | 44 (51.8%) | 40 (47.1%) | 15 (17.6%) | 10 (11.8%) | 20 (23.5%) |
| `3-Cl` (False Positives) | 42 | 12 (28.6%) | 10 (23.8%) | 2 (4.8%) | **0 (0.0%)** | **0 (0.0%)** |
| `3-KNa` (False Positives) | 35 | 9 (25.7%) | 8 (22.9%) | 5 (14.3%) | 6 (17.1%) | **1 (2.9%)** |
| `3-K` (False Positives) | 20 | 5 (25.0%) | 4 (20.0%) | 2 (10.0%) | 1 (5.0%) | **0 (0.0%)** |
| `3-P` (False Positives) | 9 | 3 (33.3%) | 4 (44.4%) | 1 (11.1%) | **0 (0.0%)** | **0 (0.0%)** |
| `Type 1` Water Ice (False Positives) | 99 | 15 (15.2%) | 8 (8.1%) | 7 (7.1%) | 3 (3.0%) | **0 (0.0%)** |
| `Type 2` Organic Ice (False Positives) | 99 | 16 (16.2%) | 10 (10.1%) | 7 (7.1%) | 5 (5.1%) | **0 (0.0%)** |
| `Type 4`, `5`, `5-Na`, `Noise` (FP) | 268 | 4 | 0 | 2 | 0 | **0 (0.0%)** |

---

## 3. Full Test Set Breakdown (`18,305` Unique SCLKs — [`per_class_breakdown.csv`](data/results/per_class_breakdown.csv))

| Catalogue Subclass | Total Test Spectra | Stage-1 Bypassed | Sent to `gemini-4-argon` | Predicted `3-OH` | Predicted `non-3-OH` | Subclass Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`3-OH` (Sodium Hydroxide — Target)** | **41** | **0** | **41** | **37 (TP)** | **4 (FN)** | **`90.24%`** |
| `Noise` | 12,074 | 3,848 | 8,226 | 0 | 12,074 | **`100.00%`** |
| `1` (Pure Water Ice) | 4,490 | 287 | 4,203 | 3 | 4,487 | **`99.93%`** |
| `2` (Organic-Enriched Ice) | 1,014 | 55 | 959 | 0 | 1,014 | **`100.00%`** |
| `4` (Iron/Silicate Minerals) | 262 | 2 | 260 | 1 | 261 | **`99.62%`** |
| `5` (High-Mass / Other) | 186 | 9 | 177 | 0 | 186 | **`100.00%`** |
| `3-Car` (Sodium Carbonate/Formate) | 87 | 0 | 87 | 21 | 66 | `75.86%` |
| `3-Cl` (Sodium Chloride) | 45 | 0 | 45 | 0 | 45 | **`100.00%`** |
| `3-KNa` (Potassium-Sodium Salt) | 40 | 0 | 40 | 2 | 38 | **`95.00%`** |
| `5-Na` (Pure Elemental Sodium) | 36 | 0 | 36 | 0 | 36 | **`100.00%`** |
| `3-K` (Potassium Salt) | 21 | 0 | 21 | 0 | 21 | **`100.00%`** |
| `3-P` (Sodium Phosphate) | 9 | 0 | 9 | 0 | 9 | **`100.00%`** |
| **Total** | **18,305** | **4,202 (`22.95%`)** | **14,105** | **64 (`37` TP, `27` FP)** | **18,241** | **`99.83%`** |

---

## 4. Cost & Token Telemetry Summary (per 25,000 Spectra)

Because each request shares an 18–21 exemplar prefix (`~25.6K–26.4K` tokens) followed by 1 target spectrum (`~2.9K` tokens), **86%–91% of input tokens hit Vertex AI's automatic implicit cache** (`90%` discount on `gemini-3.5-flash` vs. **`95%` discount** on `gemini-4-argon`), and `gemini-4-argon` uses **38%–77% fewer thinking/output tokens** per spectrum:

| Run Configuration (25,000 Spectra) | Total Input Tokens | Cached Input Share | Output + Thinking Tokens | Batch Cost (with Implicit Cache) | Batch Cost (with 23% Stage-1 Pre-Filter) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `gemini-3.5-flash` (HITL) | `729.2M` | `85.8%` | `13.11M` | `$230.30` | `$177.45` |
| `gemini-3.5-flash` (AI-Only) | `705.5M` | `90.8%` | `24.22M` | `$253.88` | `$195.61` |
| `gemini-4-argon` (HITL, Intro Rate) | `740.5M` | `87.3%` | `9.09M` | `$204.09` | `$157.25` |
| `gemini-4-argon` (AI-Only, Intro Rate) | `718.5M` | `87.0%` | `5.67M` | **`$184.27`** | **`$141.98`** |
| **`cda_3oh_agent` Actual Run (`18,305` Test Spectra, `14,105` VLM)** | `405.4M` | `87.0%` | `3.45M` | — | **`$112.49`** |

---

## 5. Key Artifacts in This Branch

- **Agent & Multi-Agent Prompt Optimization**:
  - [`cda_3oh_agent/refined_prompt.txt`](refined_prompt.txt): Final Referee-synthesized binary `3-OH` system instruction.
  - [`cda_3oh_agent/data/results/academic_research_proposal.md`](data/results/academic_research_proposal.md): `AcademicResearchAgent` proposal grounded in Postberg et al. (2008, 2009, 2018, 2023) & Hillier et al. (2007).
  - [`cda_3oh_agent/data/results/academic_critic_report.md`](data/results/academic_critic_report.md): `AcademicCriticAgent` critique of mass-overlap failure modes (`3-Car`, `3-KNa`, `3-Cl`).
  - [`cda_3oh_agent/data/results/academic_referee_decision.md`](data/results/academic_referee_decision.md): `AcademicRefereeAgent` synthesis and decision rules.
  - [`cda_3oh_agent/data/input/examples/cached_examples.jsonl`](data/input/examples/cached_examples.jsonl): 18 stratified few-shot exemplars annotated by `gemini-4-argon`.
- **Full `18,305`-Spectrum Test Set Outputs**:
  - [`cda_3oh_agent/data/results/results.csv`](data/results/results.csv) & [`results.parquet`](data/results/results.parquet)
  - [`cda_3oh_agent/data/results/per_class_breakdown.csv`](data/results/per_class_breakdown.csv)
  - [`cda_3oh_agent/data/results/metrics_3oh_agent.json`](data/results/metrics_3oh_agent.json)
  - [`cda_3oh_agent/data/results/confusion_matrix.png`](data/results/confusion_matrix.png)
- **Notebook Study & 24,829 Unlabelled Inference Runs (`Notebooks/3oh_classification/`)**:
  - [`Notebooks/3oh_classification/study_results_3oh/`](../Notebooks/3oh_classification/study_results_3oh/): `gemini-3.5-flash` HITL study results (`707` spectra).
  - [`Notebooks/3oh_classification/study_results_3oh_argon/`](../Notebooks/3oh_classification/study_results_3oh_argon/): `gemini-4-argon` HITL study results (`707` spectra).
  - [`Notebooks/3oh_classification/study_results_3oh_ai_only/`](../Notebooks/3oh_classification/study_results_3oh_ai_only/): `gemini-3.5-flash` AI-Only study results (`698` spectra).
  - [`Notebooks/3oh_classification/study_results_3oh_ai_only_argon/`](../Notebooks/3oh_classification/study_results_3oh_ai_only_argon/): `gemini-4-argon` AI-Only study results (`698` spectra).
  - [`Notebooks/3oh_classification/inference_results_3oh/`](../Notebooks/3oh_classification/inference_results_3oh/): `gemini-3.5-flash` HITL unlabelled inference (`170` `3-OH` candidates).
  - [`Notebooks/3oh_classification/inference_results_3oh_ai_only/`](../Notebooks/3oh_classification/inference_results_3oh_ai_only/): `gemini-3.5-flash` AI-Only unlabelled inference (`92` `3-OH` candidates).
  - [`Notebooks/3oh_classification/inference_results_3oh_argon/`](../Notebooks/3oh_classification/inference_results_3oh_argon/): `gemini-4-argon` HITL unlabelled inference (`228` `3-OH` candidates) + [`candidates_3oh_comparison.csv`](../Notebooks/3oh_classification/inference_results_3oh_argon/candidates_3oh_comparison.csv) across all 282 union candidates.
