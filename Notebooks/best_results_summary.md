# Cassini CDA Dust Analyzer - Best Notebook Results & Benchmarks

This document summarizes the key performance metrics, ablation study results, and benchmark findings from the experiments in the `Notebooks/` directory.

---

## 1. Overall Highest Accuracy by Sub-domain / Signal Bin
From the **$Q_I$ Signal Amplitude Ablation Study** ([06_qi_ablation_study.ipynb](file:///Users/turanbulmus/Documents/CDA%20Dust%20Analyzer/Notebooks/06_qi_ablation_study.ipynb)):
- **Peak Overall Accuracy: 85.85%** achieved on dust spectra within the log $Q_I$ signal amplitude range of `[-13.20, -12.80)`.
- **$Q_I$ Amplitude Performance Breakdown**:
  - `[-13.20, -12.80)`: **85.85%** (Peak accuracy)
  - `[-13.60, -13.20)`: **84.01%**
  - `[-14.00, -13.60)`: **83.27%**
  - `[-12.80, -12.40)`: **80.51%**
  - `[-12.40, -12.00)`: **77.39%**

---

## 2. Few-Shot / Incremental Learning Performance
From [02_ablation_study_incremental.ipynb](file:///Users/turanbulmus/Documents/CDA%20Dust%20Analyzer/Notebooks/02_ablation_study_incremental.ipynb):
- Evaluated performance scaling as the shot count $k$ increased from $1$ to $24$ over 544 test spectra:
  - **$k=1$ (1-shot learning)**: **70.77%** accuracy
  - **$k=24$ (24-shot learning)**: **83.09%** accuracy *(+12.32% improvement with additional in-context examples)*

---

## 3. Rare Class 3-P (Pure Ice / Peak 3) Targeted Ablation Results
Multiple prompt strategies were tested in notebooks `07` through `10` to address class imbalance for the rare Class 3-P dust particles ($k=4$ shot):

| Strategy / Notebook | Overall Accuracy | Class 3-P Recall | Class 3-P Precision | Class 3-P F1-Score | Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Hybrid Prompt 3-P** <br>([07_3p_ablation_study.ipynb](file:///Users/turanbulmus/Documents/CDA%20Dust%20Analyzer/Notebooks/07_3p_ablation_study.ipynb)) | **81.98%** | **100.00%** (5/5) | **16.13%** | **27.78%** | **Best overall F1-score & 100% recall** for rare 3-P class. |
| **Self-Correcting Prompt** <br>([10_3p_ablation_study_self_correct.ipynb](file:///Users/turanbulmus/Documents/CDA%20Dust%20Analyzer/Notebooks/10_3p_ablation_study_self_correct.ipynb)) | **83.25%** | 60.00% (3/5) | 10.34% | 17.65% | **Highest overall classification accuracy** among 3-P runs. |
| **Hierarchical / Reclassified** <br>([09_3p_ablation_study_reclassified.ipynb](file:///Users/turanbulmus/Documents/CDA%20Dust%20Analyzer/Notebooks/09_3p_ablation_study_reclassified.ipynb)) | 81.34% | **100.00%** (5/5) | 9.09% | 16.67% | High sensitivity (100% recall) but higher false positives. |
| **Dual Plot Mode** <br>([08_3p_ablation_study_dual.ipynb](file:///Users/turanbulmus/Documents/CDA%20Dust%20Analyzer/Notebooks/08_3p_ablation_study_dual.ipynb)) | 81.98% | 40.00% (2/5) | 11.11% | 17.39% | Lower recall compared to single spectral view prompts. |

---

## Key Findings Summary
1. **Best In-Context Model Performance**: Scaled to **83.09%** accuracy at 24-shot learning.
2. **Best Signal Range**: Signal amplitudes in the **log $Q_I \in [-13.20, -12.80)$** range yield the cleanest features, reaching **85.85%** accuracy.
3. **Best 3-P Detection Model**: The hybrid prompt in [07_3p_ablation_study.ipynb](file:///Users/turanbulmus/Documents/CDA%20Dust%20Analyzer/Notebooks/07_3p_ablation_study.ipynb) achieved **100% sensitivity (Recall)** on the rare Class 3-P particles with a peak F1-score of **27.78%**.
