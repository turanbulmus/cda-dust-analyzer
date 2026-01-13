# CDA Dust Analyzer

**A Multimodal Classification System for Cosmic Dust Analyzer (CDA) Spectra using Gemini 3.0 Pro.**

This project implements an automated pipeline to identify rare **Class 4** dust impact events from Time-of-Flight mass spectrometry data. It leverages Google's **Gemini 3.0 Pro - Preview** Model via Vertex AI to analyze spectral plots as images, achieving high recall through visually-grounded logical reasoning.

## 🚀 Key Features

-   **Multimodal Analysis**: Converts raw spectral data into high-contrast visualizations for Gemini consumption.
-   **Automated Feature Discovery**: Includes a pipeline (`optimize_prompt.py`) that empirically derives classification rules from ground-truth samples.
-   **Global Batch Processing**: Implements a hybrid submission system (`Python` SDK + `curl`) to bypass regional model availability restrictions.
-   **High Accuracy**: Achieved **84% Recall** and **85% Accuracy** on a balanced dataset of 1800 samples.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `batch_classify.py` | **Core Pipeline.** Handles local evaluation and global batch submission. |
| `optimize_prompt.py` | **Research Tool.** Derives classification rules from data. |
| `analyze_results.py` | **Analysis.** Computes metrics from batch output. |
| `download_results.py` | **Utility.** Downloads predictions from GCS. |
| `data/` | **Data Directory.** Contains input parquet files. |
| `archive/` | **History.** Initial prompts and baseline results. |

## 🛠️ Usage

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/turanbulmus/cda-dust-analyzer.git
cd cda-dust-analyzer

# Install dependencies
pip install -r requirements.txt
```

### 2. Authentication
This project requires Google Cloud credentials.
*   **Local Evaluation**: Copy the example environment file and fill in your configuration:
    ```bash
    cp .env.example .env
    # Edit .env to set:
    # - GOOGLE_CLOUD_API_KEY
    # - GOOGLE_CLOUD_PROJECT
    # - GCS_BUCKET_NAME
    ```
*   **Batch Processing**: requires Application Default Credentials (ADC):
    ```bash
    gcloud auth application-default login
    ```

### 3. Quick Start (Local)
Run a rapid verification on 20 random samples to verify the model logic:
```bash
python batch_classify.py --local-eval
```

### 3. Automated Feature Research (Optimization)
**Recommended Step**: Run this *before* large scale batch processing to ensure the prompt is tuned to the current data.
```bash
python optimize_prompt.py
```
**What this does:**
1.  **Selects** random samples from each class.
2.  **Analyzes** them using Gemini to extract features.
3.  **Synthesizes** a new System Instruction and User Prompt.
4.  **Updates** `batch_classify.py` automatically with the new prompt.

### 4. Full Batch Processing
Submit the entire dataset (e.g., 1800 samples) to Vertex AI, wait for completion, and automatically analyze results:

If `GCS_BUCKET_NAME` is set in your `.env` file (recommended):
```bash
python batch_classify.py
```

Otherwise, specify the bucket manually:
```bash
python batch_classify.py --bucket gs://YOUR_BUCKET_NAME
```

**What this does:**
1.  **Generates** a JSONL input file from the dataset.
2.  **Uploads** it to your GCS bucket.
3.  **Submits** a Batch Prediction Job to the Global Endpoint.
4.  **Polls** the job status every 30 seconds until completion.
5.  **Downloads** the predictions (`predictions.jsonl`) automatically.
6.  **Runs Analysis** (`analyze_results.py`) to generate a report and log metrics to Vertex AI Experiments.

### 5. Manual Analysis (Optional)
If you need to re-run analysis on downloaded predictions or analyze a previous run:
```bash
python analyze_results.py
```

**Sample Output:**
```text
Overall Accuracy: 85.00%
Micro F1: 0.85

Per-Class Metrics:
Class 4: Precision=0.82, Recall=0.90, F1=0.86
Class 1: Precision=0.90, Recall=0.90, F1=0.90
Class Noise: Precision=0.80, Recall=0.75, F1=0.77

Confusion Matrix (Rows=True, Cols=Pred):
Labels: ['4', '1', 'Noise']
[[ 90   5   5]
 [  5  90   5]
 [ 15  10  75]]
```



## 🧠 Classification Logic

The system identifies **Class 4** events by distinguishing them from specific distractor classes. Plots are generated using **Logarithmic Scale** to visualize full dynamic range.

1.  **Class 4 (Target)**: Characterized by:
    *   **Mid-Range Peaks**: Distinct peaks (e.g., at X=200, 320) rising available baseline.
    *   **Repeating Patterns**: Periodic vertical structures or repeating motifs in the mid-range (even if messy).
2.  **Class 1 (Distractor)**: Defined by a single **Early Spike** (X=10-20) with a quiet or featureless mid-range.
3.  **Noise (Distractor)**: Defined by a sharp start spike (X~15) followed by a **Wall of Static** or completely featureless baseline noise throughout the mid-range.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
