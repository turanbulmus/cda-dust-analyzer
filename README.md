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
Accuracy: 84.72%
Recall: 0.84
Precision: 0.41
Confusion Matrix:
TP: 169 | FN: 31
FP: 244 | TN: 1356
```

## 🧠 Classification Logic

The system identifies **Class 4** events based on a **Central Signal Complex**:
1.  **Target Zone (X=180-400)**: Must contain a distinct, high-amplitude features (jagged peaks, double-peaks).
2.  **Rejection Criteria**:
    *   **Early Spike Only**: Strong signal at X<50 with a quiet target zone.
    *   **Wall of Static**: Continuous noise across the entire spectrum.
    *   **Weak Signals**: Faint bumps in the target zone are ignored.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
