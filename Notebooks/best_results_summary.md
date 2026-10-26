# Cassini CDA Dust Analyzer - Best Notebook & Agent Results Summary

This document summarizes the key performance metrics, ablation study results, and Stage-1 / Stage-2 benchmark findings across `Notebooks/`, `cda_dust_agent/`, and `cda_3oh_agent/`.

---

## 1. Overall Highest Multiclass Accuracy by Sub-domain / Signal Bin
From the **$Q_I$ Signal Amplitude Ablation Study** ([06_qi_ablation_study.ipynb](06_qi_ablation_study.ipynb)):
- **Peak Sub-domain Accuracy: 85.85%** achieved on dust spectra within the log $Q_I$ signal amplitude range of `[-13.20, -12.80)`.
- **$Q_I$ Amplitude Performance Breakdown**:
  - `[-13.20, -12.80)`: **85.85%** (Peak accuracy)
  - `[-13.60, -13.20)`: **84.01%**
  - `[-14.00, -13.60)`: **83.27%**
  - `[-12.80, -12.40)`: **80.51%**
  - `[-12.40, -12.00)`: **77.39%**
- **Multi-Agent 8-Class Prompt + Stage-1 Noise Pre-Filter** ([agent/reports/multi_agent_academic_analysis.md](../agent/reports/multi_agent_academic_analysis.md)):
  - Reached **87.19% overall accuracy** (**87.57% weighted F1**) across 2,000 test spectra (`96.69%` Class 1 precision, `93.55%` Noise precision, `99.16%` Noise recall).

---

## 2. Few-Shot / Incremental Learning Performance
From [02_ablation_study_incremental.ipynb](02_ablation_study_incremental.ipynb):
- Evaluated performance scaling as the shot count $k$ increased from $1$ to $24$ over 544 test spectra:
  - **$k=1$ (1-shot learning)**: **70.77%** accuracy
  - **$k=24$ (24-shot learning)**: **83.09%** accuracy *(+12.32% improvement with additional in-context examples)*

---

## 3. Rare Class `3-P` Stage-1 Ablation Results (Raw Time-of-Flight Plots)
Multiple Stage-1 prompt strategies were tested in notebooks `07` through `10` to address extreme class imbalance for rare Class `3-P` dust particles ($k=4$ shot):

| Strategy / Notebook | Overall Accuracy | Class 3-P Recall | Class 3-P Precision | Class 3-P F1-Score | Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Hybrid Prompt 3-P** <br>([07_3p_ablation_study.ipynb](07_3p_ablation_study.ipynb)) | **81.98%** | **100.00%** (5/5) | **16.13%** | **27.78%** | **Best Stage-1 F1-score & 100% recall** for rare 3-P class on raw TOF plots. |
| **Self-Correcting Prompt** <br>([10_3p_ablation_study_self_correct.ipynb](10_3p_ablation_study_self_correct.ipynb)) | **83.25%** | 60.00% (3/5) | 10.34% | 17.65% | **Highest overall classification accuracy** among Stage-1 3-P runs. |
| **Hierarchical / Reclassified** <br>([09_3p_ablation_study_reclassified.ipynb](09_3p_ablation_study_reclassified.ipynb)) | 81.34% | **100.00%** (5/5) | 9.09% | 16.67% | High sensitivity (100% recall) but higher false positives. |
| **Dual Plot Mode** <br>([08_3p_ablation_study_dual.ipynb](08_3p_ablation_study_dual.ipynb)) | 81.98% | 40.00% (2/5) | 11.11% | 17.39% | Lower recall compared to single spectral view prompts. |

---

## 4. Binary `3-OH` Classification & Stage-2 Neuro-Symbolic Adjudication (`3oh_classification/15`–`17` & `cda_3oh_agent`)

In [15_3oh_few_shot_study.ipynb](3oh_classification/15_3oh_few_shot_study.ipynb), [16_3oh_inference.ipynb](3oh_classification/16_3oh_inference.ipynb), [17_3oh_progress_monitor.ipynb](3oh_classification/17_3oh_progress_monitor.ipynb), and [`cda_3oh_agent/`](../cda_3oh_agent/README.md), we evaluated `gemini-3.5-flash` vs. `gemini-4-argon`, optimized the binary `3-OH` prompt via the 3-agent `Research` $\rightarrow$ `Critic` $\rightarrow$ `Referee` loop, and introduced **Stage-2 Two-Anchor Mass-Calibrated Adjudication** (`23.0 u` $\text{Na}^+$ and `63.0 u` $\text{Na}^+(\text{NaOH})$ anchors):

### 4.1 Held-Out Study Benchmark (`698`–`707` Labelled Spectra)

| Configuration | Model | Study $N$ | Accuracy | `3-OH` Recall | `3-OH` Precision | `3-OH` F1 | Macro F1 | False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HITL Prompt (9 pos + 9 neg) | `gemini-3.5-flash` | 707 | `83.95%` | `90.24%` (37/41) | `25.52%` | `0.3978` | `0.6531` | 108 |
| HITL Prompt (9 pos + 9 neg) | `gemini-4-argon` | 707 | `87.41%` | `92.68%` (38/41) | `31.15%` | `0.4663` | `0.6976` | 84 |
| AI-Only Literature Prompt (18-shot) | `gemini-3.5-flash` | 698 | `93.84%` | **`95.12%` (39/41)** | `48.75%` | `0.6446` | `0.8065` | 41 |
| AI-Only Literature Prompt (18-shot) | `gemini-4-argon` | 698 | `95.99%` | `92.68%` (38/41) | `60.32%` | `0.7308` | `0.8545` | 25 |
| **`cda_3oh_agent` Stage-1 (Referee Prompt)** | **`gemini-4-argon`** | **698** | **`96.42%`** | **`90.24%` (37/41)** | **`63.79%`** | **`0.7475`** | **`0.8641`** | **21** |

### 4.2 Full `18,305`-Spectrum Labelled Archive Progression (Stage-1 vs. Stage-2)

| Pipeline Stage (`18,305` Labelled Archive, `41` True `3-OH`) | Overall Accuracy | `3-OH` Recall (`TP / 41`) | `3-OH` Precision | `3-OH` F1 | Total FPs (`/ 18,264`) | `3-Car` FPs (`/ 87`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage-1 VLM Only (`gemini-4-argon` + Referee Prompt)** | `99.83%` | **`90.24%` (`37/41`)** | `57.81%` | `0.7048` | `27` | `21` |
| **Stage-1 + Stage-2 Specialist VLM Only (Dual-Panel Mass Plot)** | `99.86%` | **`90.24%` (`37/41`)** | `62.71%` | `0.7400` | `22` | `16` |
| **Stage-1 + Hybrid Neuro-Symbolic Stage-2 (`passes_129_veto`)** | `99.93%` | **`85.37%` (`35/41`)** | **`85.37%` (`+27.6 pp`)** | **`0.8537`** | **`6` (`-77.8%`)** | **`5` (`-76.2%`)** |
| **Stage-1 + Strict Neuro-Symbolic Stage-2 (`passes_strict_veto`)** | **`99.94%`** | `80.49%` (`33/41`) | **`91.67%` (`+33.9 pp`)** | **`0.8571`** | **`3` (`-88.9%`)** | **`3` (`-85.7%`)** |

- **Unlabelled `24,829`-Spectrum Archive Discovery** ([candidates_3oh_adjudicated.csv](3oh_classification/inference_results_3oh_argon/candidates_3oh_adjudicated.csv)): Across `282` union Stage-1 candidates, Stage-2 adjudication confirmed **`93` ultra-pure Hybrid `3-OH` grains** (`75` Tier-1 multi-model consensus + `18` Tier-2 single-model), `72` Tier-3 borderline candidates, and rejected `117` Tier-4 false positives (primarily `3-Car` carbonate grains with $129\text{ u}$ features).

---

## 5. Stage-2 Leak-Free Blind `3-P` Adjudication (`377` Stage-1 Batch FPs from $N=17,146$ Archive)

To resolve the Stage-1 `3-P` precision bottleneck (`16.13%` on the 544-sample study set and `1.31%` [`5/382`, `377` FPs] on the full `17,146`-spectrum batch run `cda-batch-20260707`), we evaluated the two-anchor mass-calibrated Stage-2 pipeline across all `377` Stage-1 batch false positives, all `8` held-out catalogued `3-P` grains, and the out-of-distribution mixed chloride-phosphate grain `SCLK 1511735208` ([metrics_3p_stage2_comparison.json](../cda_3oh_agent/data/results/metrics_3p_stage2_comparison.json)):

| `3-P` Pipeline Stage ($N = 17,146$ Batch Run) | Canonical `3-P` Recall (`TP / 8`) | Total Recall incl. Mixed `SCLK 1511735208` (`TP / 9`) | Stage-1 Batch FPs Remaining (`/ 377`) | Full-Batch Precision (`5` Benchmark `3-P`) | `SCLK 1511735208` Disposition |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage-1 Raw-Image VLM (`cda-batch-20260707`)** | **`100.0%` (`8/8`)** | **`100.0%` (`9/9`)** | `377` (`100%`) | `1.31%` (`5/382`) | Flagged (`1` of `36` in wild) |
| **1D 7-Parameter Bounding Box (Strict Leave-One-Out CV)** | `80.0%` (`8/10` all) | `72.7%` (`8/11` all) | `7` (`-98.1%`) | `41.67%` (`5/12`) | Fails LOOCV (`p81=0.292 > 0.086`) |
| **Tier-2 Loose Physical Sign Screen (`diff_125_129 >= 0` & Triplet $\ge 0.04$)** | **`100.0%` (`8/8`)** | **`100.0%` (`9/9`)** | `61` (`-83.8%`) / `26` (`-93.1%` w/ `p81<=0.30`) | `7.58%`–`16.13%` | **Retained (`diff_125_129 = +0.0552`)** |
| **Tier-1 Blind Stage-2 Specialist VLM (`gemini-4-argon`, Zero Threshold Leakage)** | **`100.0%` (`8/8`)** | **`88.9%` (`8/9`)** | **`0 / 377` (`-100.0%`)** | **`100.0%` (`5/5`, `F1=1.00`)** | Filtered as `3-Cl` (`p81=0.292` mixed grain) |

---

## Key Findings Summary
1. **Best Multiclass Performance**: Scaled from **83.09%** at 24-shot to **87.19% accuracy (87.57% weighted F1)** across 2,000 test spectra with the multi-agent academic prompt and Stage-1 noise pre-filter.
2. **Best Signal Range**: Signal amplitudes in **log $Q_I \in [-13.20, -12.80)$** yield the cleanest spectral envelopes, reaching **85.85%** accuracy in single-stage ablation runs.
3. **Best `3-OH` Discovery Pipeline**: Combining `gemini-4-argon` Stage-1 screening (`90.24%` recall across `18,305` labelled spectra) with Two-Anchor Mass-Calibrated Stage-2 Neuro-Symbolic Adjudication boosts `3-OH` precision from `57.81%` $\rightarrow$ **`85.37%` (`F1 = 0.8537`)** under the $129\text{ u}$ carbonate veto and **`91.67%` (`F1 = 0.8571`)** under the strict carbonate + water-comb veto, identifying **`93` hybrid-confirmed new `3-OH` grains** in the unlabelled archive.
4. **Best `3-P` Detection Pipeline**: Pairing Stage-1 VLM recall (`100%`) with the leak-free Stage-2 Dual-Panel Mass-Calibrated `gemini-4-argon` Specialist VLM eliminates **`100%` (`0/377`) of Stage-1 batch false positives** while retaining **`100%` (`8/8`) of canonical catalogued `3-P` grains** (jumping full-batch `3-P` precision from `1.31%` $\rightarrow$ **`100.0%`** and F1 from `0.0258` $\rightarrow$ **`1.0000`**), while the parameter-free Tier-2 physical sign screen (`diff_125_129 >= 0`) retains **`100%` (`9/9`)** including mixed chloride-phosphate grains.
