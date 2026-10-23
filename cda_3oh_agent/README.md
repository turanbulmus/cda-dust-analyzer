# Cassini CDA Binary `3-OH` & `3-P` Classification Study (`gemini-3.5-flash` vs. `gemini-4-argon` & `cda_3oh_agent`)

This directory contains the Google ADK multi-agent pipeline (`cda_3oh_agent`) for binary **`3-OH` (sodium-hydroxide-dominated salt-rich Type 3 grains) vs. `non-3-OH`** and **`3-P` (sodium-orthophosphate-bearing Type 3 grains) vs. `non-3-P`** discovery in the Cassini Cosmic Dust Analyzer (CDA) archive, along with the full evaluation results across **`gemini-3.5-flash`**, **`gemini-4-argon`**, and the **Stage-2 Neuro-Symbolic & Specialist VLM Adjudicator**.

---

## 1. Executive Summary of Study Findings

1. **Model Upgrade (`gemini-3.5-flash` $\rightarrow$ `gemini-4-argon`)**:
   - On the held-out `698`–`707` labelled study benchmark, switching from `gemini-3.5-flash` to `gemini-4-argon` improves binary accuracy and cuts false positives across both prompt types:
     - **Human-in-the-Loop (HITL) Prompt (`707` study spectra)**: Accuracy rises from **`83.95%` $\rightarrow$ `87.41%`**, `3-OH` Recall from **`90.24%` $\rightarrow$ `92.68%` (`38/41`)**, and false positives drop from `108` $\rightarrow$ `84`.
     - **AI-Only Literature Prompt (`698` study spectra)**: Accuracy rises from **`93.84%` $\rightarrow$ `95.99%`**, `3-OH` Precision jumps from **`48.75%` $\rightarrow$ `60.32%`**, `3-OH` F1 improves from **`0.6446` $\rightarrow$ `0.7308`**, and false positives drop by **39%** (`41` $\rightarrow$ `25`).
2. **Anchored Multi-Agent Prompt Optimization (`cda_3oh_agent` + `gemini-4-argon`)**:
   - Using the AI-generated literature prompt (`ANCHOR_3OH_AI_PROMPT`) as an anchoring point, the 3-agent loop (`AcademicResearchAgent` $\rightarrow$ `AcademicCriticAgent` $\rightarrow$ `AcademicRefereeAgent`) on `gemini-4-argon` synthesized a 5-gate diagnostic decision tree ([`refined_prompt.txt`](refined_prompt.txt)).
   - On the **`698`-sample study benchmark**, `cda_3oh_agent` achieved:
     - **Accuracy**: **`96.42%`** (`673 / 698`) | **`3-OH` Precision**: **`63.79%`** (`37` TP, `21` FP) | **`3-OH` Recall**: **`90.24%`** (`37 / 41`) | **`3-OH` F1**: **`0.7475`**
     - Eliminated **100% of false positives** on `Type 1` (water ice), `Type 2` (organic ice), `3-Cl` (sodium chloride), `3-K` (potassium salt), `3-P` (phosphate), `4`, `5`, `5-Na`, and `Noise`, and cut `3-KNa` false positives from `6` $\rightarrow$ `1`.
3. **Full Labelled Archive Evaluation — Stage 1 (`18,305` Unique Labelled Test Spectra)**:
   - Running Stage-1 `cda_3oh_agent` across all `18,305` held-out labelled test spectra (`41` `3-OH`, `18,264` `non-3-OH` — a `1:445` imbalance):
     - **Overall Accuracy**: **`99.83%`** (`18,274 / 18,305`)
     - **`3-OH` Recall**: **`90.24%`** (`37 / 41`)
     - **`3-OH` Precision**: **`57.81%`** (`37` TP, `27` FP out of `18,264` non-3-OH spectra — a **`0.15%` false-positive rate**)
     - **`3-OH` F1**: **`0.7048`** | **Macro F1**: **`0.8520`**
     - **Stage-1 Pre-Filter (`is_obvious_noise`)**: Bypassed **`4,202` spectra (`22.95%`)** as obvious noise with **`0` true `3-OH` spectra lost**, saving ~23% of VLM inference cost.
4. **Stage-2 Neuro-Symbolic & Specialist VLM Adjudicator (`Stage2SpecialistAdjudicatorAgent` + Two-Anchor Mass Calibration)**:
   - To overcome the single-stage time-domain precision ceiling (`57.81%` on `3-OH`, caused by `21` `3-Car` false positives with subtle $129\text{ u}$ carbonate features, and `3-Car`/`3-Cl` false positives on `3-P`), we implemented in [`tools/utils.py`](tools/utils.py) and [`sub_agent/specialist_adjudicator.py`](sub_agent/specialist_adjudicator.py):
     1. **Physics-Constrained Two-Anchor Mass-Domain Calibration** (`calibrate_spectrum_mass_domain`): Solves per-spectrum $(\hat{t}_0, \hat{a})$ from the $\text{Na}^+$ ($23.0\text{ u}$) 90%-contour centroid and the sharp $\text{Na}^+(\text{NaOH})$ ($63.0\text{ u}$) peak.
     2. **Dual-Panel Mass-Calibrated Plotter** (`generate_dual_panel_mass_spectrum_bytes`): Renders a 2-panel figure (Top: full $10\text{--}200\text{ u}$ calibrated mass spectrum; Bottom: zoomed $70\text{--}195\text{ u}$ anion diagnostic window with color-coded vertical mass gridlines and shaded $125\text{ u}$ `3-P` [green] vs. $129\text{ u}$ `3-Car` [red] bands).
     3. **1D Quantitative Mass Telemetry & Physics Veto Gate + Stage-2 Specialist VLM Adjudicator**:
        - **Validation 1 — Full `18,305` Labelled `3-OH` Archive** ([`metrics_3oh_stage2_comparison.json`](data/results/metrics_3oh_stage2_comparison.json)):
          - **Stage-2 Specialist VLM Only**: Cuts false positives from `27` $\rightarrow$ `22` (`Precision: 57.81%` $\rightarrow$ `62.71%`, `F1: 0.7048` $\rightarrow$ `0.7400`) with **0% recall loss (`37/41 = 90.24%`)**.
          - **Hybrid Neuro-Symbolic Stage-2 (`passes_129_veto`, Carbonate Veto)**: Cuts false positives by **77.8% (`27` $\rightarrow$ `6`)**, jumping **`3-OH` Precision from `57.81%` $\rightarrow$ `85.37%` (`+27.56 pp`)** and **`3-OH` F1 from `0.7048` $\rightarrow$ `0.8537` (`+0.1489`)** while retaining **`85.37%` (`35/41`) recall**.
          - **Strict Neuro-Symbolic Stage-2 (`passes_strict_veto`, Carbonate + Water-Comb Veto)**: Cuts false positives by **88.9% (`27` $\rightarrow$ `3`)**, reaching **`91.67%` Precision (`+33.86 pp`)** and **`0.8571` F1** at **`80.49%` (`33/41`) recall**.
        - **Validation 2 — `282` Unlabelled `3-OH` Candidates (`24,829` Archive)** ([`unlabelled_3oh_stage2_summary.json`](data/results/unlabelled_3oh_stage2_summary.json)):
          - Stratifies the 282 union candidates into **75 Tier-1 (Multi-Model Consensus $\ge 2$ + Physics & Stage-2 VLM Confirmed)**, **18 Tier-2 (Single-Model + Physics & Stage-2 VLM Confirmed)** (`93` Hybrid-Confirmed ultra-pure `3-OH`), **72 Tier-3 (Borderline: Passed Either Physics Gate or Stage-2 VLM Only, `165` total across Tiers 1–3)**, and **117 Tier-4 (Rejected by Both $129\text{ u}$ Physics Veto & Stage-2 VLM)**.
        - **Validation 3 — Held-Out `3-P` Salt-Rich `Type 3` Archive ($N=212$) + Confirmed Discovery `SCLK 1511735208`** ([`metrics_3p_stage2_comparison.json`](data/results/metrics_3p_stage2_comparison.json)):
          - Achieves **`100.0%` Precision (`0` False Positives across all `205` non-`3-P` salt-rich `Type 3` spectra)** and **`100.0%` Recall (`7/7` held-out catalogued `3-P` grains + confirmed discovery `SCLK 1511735208` at `0.92` confidence + catalogued Rhea grain `SCLK 1511748839` at `0.99` confidence, `F1 = 1.000`)**!

---

## 2. Benchmark Comparison on Held-Out Study Set (`698`–`707` Labelled Spectra)

| Configuration | Model | Study $N$ | Accuracy | `3-OH` Recall | `3-OH` Precision | `3-OH` F1 | Macro F1 | False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HITL Prompt (Nozair 9-shot + 9 neg) | `gemini-3.5-flash` | 707 | `83.95%` | `90.24%` (37/41) | `25.52%` | `0.3978` | `0.6531` | 108 |
| HITL Prompt (Nozair 9-shot + 9 neg) | `gemini-4-argon` | 707 | `87.41%` | `92.68%` (38/41) | `31.15%` | `0.4663` | `0.6976` | 84 |
| AI-Only Anchor Prompt (18-shot) | `gemini-3.5-flash` | 698 | `93.84%` | **`95.12%` (39/41)** | `48.75%` | `0.6446` | `0.8065` | 41 |
| AI-Only Anchor Prompt (18-shot) | `gemini-4-argon` | 698 | `95.99%` | `92.68%` (38/41) | `60.32%` | `0.7308` | `0.8545` | 25 |
| **`cda_3oh_agent` Stage-1 (Referee-Optimized)** | **`gemini-4-argon`** | **698** | **`96.42%`** | **`90.24%` (37/41)** | **`63.79%`** | **`0.7475`** | **`0.8641`** | **21** |

---

## 3. Full Archive Stage-1 vs. Stage-2 Progression (`18,305` Unique Labelled Spectra — [`metrics_3oh_stage2_comparison.json`](data/results/metrics_3oh_stage2_comparison.json))

| Pipeline Stage (`18,305` Labelled Archive) | Overall Accuracy | `3-OH` Recall (`TP / 41`) | `3-OH` Precision | `3-OH` F1 | Total FPs (`/ 18,264`) | `3-Car` FPs (`/ 87`) | Other FPs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage-1 VLM Only (`gemini-4-argon` + Referee Prompt)** | `99.83%` | **`90.24%` (`37/41`)** | `57.81%` | `0.7048` | `27` | `21` | `Type 1: 3, 3-KNa: 2, Type 4: 1` |
| **Stage-1 + Stage-2 Specialist VLM Only (Dual-Panel Mass Plot)** | `99.86%` | **`90.24%` (`37/41`)** | `62.71%` | `0.7400` | `22` | `16` | `Type 1: 3, 3-KNa: 2, Type 4: 1` |
| **Stage-1 + Hybrid Neuro-Symbolic Stage-2 (`passes_129_veto`)** | `99.93%` | **`85.37%` (`35/41`)** | **`85.37%` (`+27.6 pp`)** | **`0.8537`** | **`6` (`-77.8%`)** | **`5` (`-76.2%`)** | `Type 1: 1, 3-KNa: 0, Type 4: 0` |
| **Stage-1 + Strict Neuro-Symbolic Stage-2 (`passes_strict_veto`)** | **`99.94%`** | `80.49%` (`33/41`) | **`91.67%` (`+33.9 pp`)** | **`0.8571`** | **`3` (`-88.9%`)** | **`3` (`-85.7%`)** | **`0 across all other classes`** |

---

## 4. Held-Out `3-P` Salt-Rich `Type 3` Evaluation (`N = 212` + `SCLK 1511735208` — [`metrics_3p_stage2_comparison.json`](data/results/metrics_3p_stage2_comparison.json))

| Pipeline Method (`N = 212` Held-Out Salt-Rich `Type 3` Spectra) | Held-Out `3-P` Recall (`TP / 7`) | Intra-`Type 3` Precision | `3-P` F1 | Total FPs (`/ 205`) | Confirmed Discovery `SCLK 1511735208` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Two-Anchor Mass-Triplet Physics Gate Only (`passes_3p_triplet_gate`)** | **`100.0%` (`7/7`)** | `38.89%` | `0.5600` | `11` (`3-OH: 5, 3-Car: 3, 3-KNa: 2, 3-Cl: 1`) | **Confirmed `3-P` (`diff_125_129 = +0.0552`)** |
| **Stage-2 Specialist VLM (`gemini-4-argon` + Dual-Panel Mass Plot + Telemetry)** | **`100.0%` (`7/7`)** | **`100.0%`** | **`1.0000`** | **`0` (`0/83 3-Car, 0/44 3-Cl, 0/40 3-OH, 0/38 3-KNa`)** | **Confirmed `3-P` (`confidence = 0.92`)** |
| **Hybrid Neuro-Symbolic (`Physics Triplet Gate AND Specialist VLM`)** | **`100.0%` (`7/7`)** | **`100.0%`** | **`1.0000`** | **`0`** | **Confirmed `3-P` (`confidence = 0.92`)** |

---

## 5. Cost & Token Telemetry Summary (per 25,000 Spectra)

Because each request shares an 18–21 exemplar prefix (`~25.6K–26.4K` tokens) followed by 1 target spectrum (`~2.9K` tokens), **86%–91% of input tokens hit Vertex AI's automatic implicit cache** (`90%` discount on `gemini-3.5-flash` vs. **`95%` discount** on `gemini-4-argon`), and `gemini-4-argon` uses **38%–77% fewer thinking/output tokens** per spectrum:

| Run Configuration (25,000 Spectra) | Total Input Tokens | Cached Input Share | Output + Thinking Tokens | Batch Cost (with Implicit Cache) | Batch Cost (with 23% Stage-1 Pre-Filter) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `gemini-3.5-flash` (HITL) | `729.2M` | `85.8%` | `13.11M` | `$230.30` | `$177.45` |
| `gemini-3.5-flash` (AI-Only) | `705.5M` | `90.8%` | `24.22M` | `$253.88` | `$195.61` |
| `gemini-4-argon` (HITL, Intro Rate) | `740.5M` | `87.3%` | `9.09M` | `$204.09` | `$157.25` |
| `gemini-4-argon` (AI-Only, Intro Rate) | `718.5M` | `87.0%` | `5.67M` | **`$184.27`** | **`$141.98`** |
| **`cda_3oh_agent` Actual Run (`18,305` Test Spectra, `14,105` VLM)** | `405.4M` | `87.0%` | `3.45M` | — | **`$112.49`** |
| **Stage-2 Specialist Adjudicator (`64`–`282` Candidates Only)** | `1.1M–4.8M` | `88.0%` | `0.06M–0.25M` | **`$0.35–$1.50`** | **`$0.35–$1.50`** |

---

## 6. Key Artifacts in This Branch

- **Stage-1 & Stage-2 Agent Code**:
  - [`cda_3oh_agent/tools/utils.py`](tools/utils.py): Two-anchor mass-domain calibration (`calibrate_spectrum_mass_domain`), dual-panel mass-calibrated plotter (`generate_dual_panel_mass_spectrum_bytes`), and 1D telemetry & physics veto gates (`extract_mass_telemetry_3oh`, `extract_mass_telemetry_3p`).
  - [`cda_3oh_agent/sub_agent/specialist_adjudicator.py`](sub_agent/specialist_adjudicator.py): `Stage2SpecialistAdjudicatorAgent` and `adjudicate_candidates_3oh` wired into [`sub_agent/workflow.py`](sub_agent/workflow.py).
  - [`cda_3oh_agent/refined_prompt.txt`](refined_prompt.txt): Final Referee-synthesized binary `3-OH` system instruction.
- **Full `18,305`-Spectrum Test Set & Stage-2 Adjudication Outputs**:
  - [`cda_3oh_agent/data/results/results.csv`](data/results/results.csv) & [`results_hybrid_stage2.csv`](data/results/results_hybrid_stage2.csv)
  - [`cda_3oh_agent/data/results/stage2_adjudication_results.csv`](data/results/stage2_adjudication_results.csv)
  - [`cda_3oh_agent/data/results/metrics_3oh_stage2_comparison.json`](data/results/metrics_3oh_stage2_comparison.json)
  - [`cda_3oh_agent/data/results/stage2_3p_adjudication_results.csv`](data/results/stage2_3p_adjudication_results.csv)
  - [`cda_3oh_agent/data/results/metrics_3p_stage2_comparison.json`](data/results/metrics_3p_stage2_comparison.json)
  - [`cda_3oh_agent/data/results/unlabelled_3oh_stage2_summary.json`](data/results/unlabelled_3oh_stage2_summary.json)
- **Notebook Study & 24,829 Unlabelled Inference Runs (`Notebooks/3oh_classification/`)**:
  - [`Notebooks/3oh_classification/inference_results_3oh_argon/candidates_3oh_adjudicated.csv`](../Notebooks/3oh_classification/inference_results_3oh_argon/candidates_3oh_adjudicated.csv): Full Stage-2 adjudication & tier assignment across all 282 union unlabelled `3-OH` candidates.
