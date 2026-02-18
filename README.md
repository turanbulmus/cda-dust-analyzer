# CDA Dust Analyzer Agent

## Overview

This project demonstrates a multi-agent system designed for the advanced classification and analysis of Cassini's Cosmic Dust Analyzer (CDA) Time-of-Flight mass spectra. The agent is built utilizing the Google Agent Development Kit (ADK) as a standalone scientific study to automate massive scientific dataset analysis. 

It handles the entire data pipeline: from fetching raw datasets dynamically via HuggingFace hub to interactive Few-Shot expert annotation, all the way through to inference processing and metric evaluations. The agent interacts with the powerful Gemini 3.0 Pro API and allows the researcher to choose between synchronous **Local Inference** and asynchronous **Vertex AI Batch Prediction**.

## Agent Details

The key features of the CDA Dust Analyzer Multi-Agent include:

| Feature | Description |
| --- | --- |
| **Interaction Type:** | Conversational & Batch |
| **Complexity:**  | Advanced |
| **Agent Type:**  | Multi Agent |
| **Components:**  | ADK Core Tools, Dynamic Pipeline Processing, Few-Shot RAG, Human-in-the-Loop Feedback |

### Architecture
```mermaid
flowchart LR
    subgraph "CDA_Dust_Analyzer_Agent (Sequential Agent)"
        Routing([Routing])
        Routing --> DataFetchAndParse([DataFetchAndParse])
        DataFetchAndParse --> FewShotAnnotation([FewShotAnnotation])
        FewShotAnnotation --> DataPrep([DataPrep])
        DataPrep --> LocalInference([LocalInference])
        LocalInference --> BatchSubmission([BatchSubmission])
        BatchSubmission --> BatchPolling([BatchPolling])
        BatchPolling --> ResultAnalysis([ResultAnalysis])
    end
```
### Key Features

* **Multi-Agent Architecture:** Utilizes a top-level `SequentialAgent` orchestrator to seamlessly string together independent data processing, polling, and execution modules.
* **Automated Data Fetching:** Built-in integration with HuggingFace Hub to dynamically download, parse, and scale incoming `.parquet` files for standardized agent consumption.
* **Interactive Human-in-the-loop (with Caching):** The `FewShotAnnotationAgent` visually iterates through subset samples, plotting regions of interest using `matplotlib` with **Logarithmic Scale Visualization (`plt.semilogy()`)**. This allows researchers to extract dynamic expert explanations that ground the LLM's multi-shot prompt, specifically focusing on dynamic range features and "Repeating Patterns". Inferences are eagerly saved to a local `.jsonl` cache, permitting instant reloading on subsequent runs without redundant manual annotation.
* **Vertex AI Batch Integration:** Asynchronous pipeline logic to bundle thousands of mass spectrometer readings into `jsonl` payloads, submitting jobs to Google Cloud, polling for completion, and automatically executing `ResultAnalysis`.
* **Dynamic Results Reporting:** Analyzes incoming JSON model structures, maps back target ID `sclk` arrays, and generates granular **Multiclass Metrics** (Precision/Recall/F1 for specific classes, rather than binary classification) alongside explicit reasoning into `results.csv`.
* **Automated Feature Research:** Includes an offline pipeline to recursively refine `SYSTEM_INSTRUCTION_TEXT` via model-driven feedback loops, ensuring that the production prompts are strictly tuned to detect structures like periodic vertical repetitions for robust Class 4 classification.
* **ADK Web GUI:** Offers a user-friendly UI development server for real-time interaction, tracking memory state, inputs, and events locally.

## Agent Setup and Installation

### Prerequisites

*   **Google Cloud Account:** Configured access to the Vertex AI API (Gemini 3.0 Pro). You must have Application Default Credentials configured.
*   **Python 3.10+:** Ensure you have Python 3.10 or a later version installed.
*   **uv:** Install the astral `uv` package manager by following the instructions on the official uv website:
    [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Project Initialization

This agent uses `uv` to manage the environment and dependencies. When you initiate an ADK command using `uv run`, the dependencies defined in `pyproject.toml` are automatically synced and invoked transparently.

## Running the Agent

You can interact with the system via the command line or the UI development server:

### CLI Interaction
```bash
# Sync dependencies and run the conversational agent via CLI
uv run adk run cda_dust_agent
```
1. **Routing:** The agent will first prompt you to choose between **local** and **batch** inference.
   - Type `local` to sample the parquet and intelligently generate inferences from Gemini synchronously.
   - Type `batch` to generate the bulk payload, submit to the Vertex AI Batch prediction queue, and poll for results asynchronously.
2. **Interactive Few-Shot Annotation:** Designed to give explicit feedback loops, the `FewShotAnnotationAgent` dynamically highlights samples. First, it will check for `cda_dust_agent/data/input/examples/cached_examples.jsonl`. If previous explanations exist, you can instantly reload them to skip manual input. Otherwise, you iteratively provide expert explanations for each class representation, guiding the multi-shot accuracy. These are eagerly cached as you go to preserve progress.
3. **Execution & Analysis:** Following annotation, either `LocalInferenceAgent` or `BatchSubmissionAgent` invokes Gemini based on your routing choice. Finally, `ResultAnalysisAgent` correlates the sample ID outputs (`sclk`) and evaluation `explanation` alongside the metrics matrix, saving the comprehensive data table contextually.

### ADK Web Interface
```bash
# Host the UI development server
uv run adk web
```
You can access the chat interface at `http://127.0.0.1:8000`.

## Folder Structure

A brief overview of the high-level structures in this repository:

- `cda_dust_agent/`: Contains the core ADK agent components.
  - `agent.py`: Defines the sequential agent graph (`RoutingAgent`, `DataFetchAndParseAgent`, `FewShotAnnotationAgent`, `DataPrepAgent`, `LocalInferenceAgent`, `BatchSubmissionAgent`, `BatchPollingAgent`, `ResultAnalysisAgent`).
  - `config.py`: Configuration settings using Pydantic, pulling from environment variables.
  - `prompts.py`: Houses the core classification definitions and user prompts.
  - `tools/`: Supportive scripts like `utils.py` for dynamic image plotting, few-shot prompt construction, and JSON structure management.
  - `data/`: Contains project data organized by pipeline stages (`raw/`, `input/`, `output/`, `results/`).
- `Notebooks/`: Contains raw experimental data exploration and parsing scratchpads.
- `scripts/legacy/`: Contains previous, standalone scripts that were used before migrating to the structured ADK framework.
- `optimize_prompt.py` (Legacy/Research): An automated script used to conduct recursive prompt engineering experiments and autowrite configurations like `SYSTEM_INSTRUCTION_TEXT` into the main application.
