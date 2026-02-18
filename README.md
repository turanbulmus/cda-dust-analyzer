# CDA Dust Analyzer Agent

This repository contains the **CDA Dust Analyzer Agent**, built with the Google Agent Development Kit (ADK). The agent is designed to classify Time-of-Flight (ToF) mass spectra into three distinct categories:
- **Class 4**: Target organic/complex spectra with mid-range structural activity.
- **Class 1**: Distractor elemental/simple spectra with a dominant early spike.
- **Noise**: Artifacts lacking chemical structure.

It interacts with the Gemini API to run inference on samples using structured outputs. It supports both **local inference** (via direct Gemini SDK calls) and **batch inference** (via Vertex AI Batch Prediction API) allowing users to choose the right path dynamically.

## Folder Structure

A brief overview of the high-level structures in this repository:

- `cda_dust_agent/`: Contains the core ADK agent components.
  - `agent.py`: Defines the sequential agent graph (`RoutingAgent`, `FewShotAnnotationAgent`, `DataPrepAgent`, `LocalInferenceAgent`, `BatchSubmissionAgent`, `BatchPollingAgent`, `ResultAnalysisAgent`).
  - `config.py`: Configuration settings using Pydantic, pulling from environment variables.
  - `prompts.py`: Houses the core classification definitions and user prompts.
  - `tools/`: Supportive scripts like `utils.py` for dynamic image plotting, few-shot prompt construction, and JSON structure management.
  - `data/`: Contains project data organized by pipeline stages.
    - `raw/`: Stores the raw dataset (e.g., `cda_sample.parquet`) for inference.
    - `input/`: Generated artifacts sent directly to Gemini models (e.g., `local_requests.jsonl` and `batch_requests.jsonl`).
    - `output/`: Local and downloaded batch model responses (e.g., `predictions.jsonl`).
    - `results/`: Processed analysis and evaluation output produced by the ResultAnalysisAgent. Specifically, `results.csv` includes the explicit `sclk` target ID, the `true_class`, the `predicted_class`, and the detailed reasoning within `explanation`.
- `scripts/legacy/`: Contains previous, standalone scripts (`batch_classify.py`, `analyze_results.py`) that were used before migrating to the structured ADK framework.

## Prerequisites

- [Google Agent Development Kit (ADK) Python SDK](https://github.com/google/agent-development-kit)
- Python 3.10+
- The `.venv` environment initialized and active.
- Configured access to the Vertex AI API (Gemini 3.0 Pro).

## Running the Agent

This agent uses the interactive ADK CLI to trace state locally. Once inside your Virtual Environment (`source .venv/bin/activate`), run the agent and interact with it via the command line or the UI:

### CLI Interaction
```bash
# Run the conversational agent via CLI
adk run cda_dust_agent
```
1. **Routing:** The agent will first prompt you to choose between **local** and **batch** inference.
   - Type `local` to sample the parquet and intelligently generate inferences from Gemini synchronously.
   - Type `batch` to generate the bulk payload, submit to the Vertex AI Batch prediction queue, and poll for results asynchronously.
2. **Interactive Few-Shot Annotation:** Designed to give explicit feedback loops, the `FewShotAnnotationAgent` dynamically highlights samples utilizing `matplotlib`. You iteratively provide expert explanations for each class representation, guiding the multi-shot accuracy.
3. **Execution & Analysis:** Following annotation, either `LocalInferenceAgent` or `BatchSubmissionAgent` invokes Gemini based on your routing choice. Finally, `ResultAnalysisAgent` correlates the sample ID outputs (`sclk`) and evaluation `explanation` alongside the metrics matrix, saving the comprehensive data table contextually.

### ADK Web Interface
```bash
# Host the UI development server
adk web
```
You can access the chat interface at `http://127.0.0.1:8000`.

## Batch Output
For batch inferences, results are uploaded to the configured GCS bucket (e.g., `gs://<your-bucket-name>/output`), and optionally downloaded locally by the `ResultAnalysisAgent` once the job succeeds.
