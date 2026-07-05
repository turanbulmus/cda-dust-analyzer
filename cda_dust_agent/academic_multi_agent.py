import os
import io
import json
import base64
import asyncio
import random
import time
import tqdm
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pypdf import PdfReader
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
import huggingface_hub

from google import genai
from google.genai import types
from google.cloud import storage
from google.cloud import aiplatform

# Load environment variables
load_dotenv('.env')

PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'turan-genai-bb')
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'turansgenaibb').replace("gs://", "")
LOCATION = "global"
MODEL_ID = 'gemini-3.5-flash'
OUTPUT_DIR = "study_results_multi_agent"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_generate_content(client, model, contents):
    try:
        return client.models.generate_content(model=model, contents=contents)
    except Exception as e:
        if "404" in str(e) or "NOT_FOUND" in str(e):
            print(f"Model {model} returned 404, falling back to gemini-2.5-flash...")
            return client.models.generate_content(model="gemini-2.5-flash", contents=contents)
        raise e

# ---------------------------------------------------------
# Baseline Prompts from 10_3p_ablation_study_self_correct.ipynb
# ---------------------------------------------------------
BASELINE_SYSTEM_INSTRUCTION_TEXT = """You are an expert Cosmic Dust Spectroscopist analyzing Cassini Cosmic Dust Analyzer (CDA) time-of-flight mass spectra.
You will be provided with images of 1D spectra plotted on a logarithmic y-axis (signal amplitude) and a linear x-axis (time-of-flight index).

---

### CRITICAL SPECTRAL BEHAVIOR & SHIFTING:
- These spectra can be shifted in time by up to 50 index points due to hardware trigger recording differences.
- Because of the non-linear mapping between time-of-flight and mass, this shift causes the peaks to visually stretch.
- Do NOT rely on absolute x-axis index positions. Instead, focus on relative shapes, relative peak sequences, and overall topography.

---

### CRITICAL DIFFERENTIATION GUIDE FOR CLASS 3 VS CLASS 3-P:
When a spectrum exhibits broad organic-like envelopes or carbon cluster humps, you must distinguish between Class 3 (General Organics) and Class 3-P (Burst-and-Trail Agglomerate):
1. **Class 3 (General Organics):** Features multiple broad, asymmetric "shark-fin" peak clusters with expanding periodicity (representing carbon series). Crucially, on the scaled [0, 1] y-axis, the valleys between clusters drop significantly lower (down to y < 0.15), and it lacks a singular early maximum (index 200) that dominates the entire spectrum by a factor of 10.
2. **Class 3-P (Burst-and-Trail Agglomerate):** Dominated by an explosive primary complex (index 190-220) which is the absolute global maximum (y = 1.0), and a distinct secondary peak at index 330-345. Crucially, past index 400 it rests on a continuous, flat "chemical noise" plateau (baseline cushion) that remains highly elevated (y ≈ 0.3 to 0.5) and never drops back to y ≈ 0, before dropping off sharply into the noise floor around index 650-750.

---

### CLASS SPECIFIC PROFILES:

#### Class Noise
- Devoid of Gaussian peaks, elemental clusters, or chemical spacing.
- Characterized by a narrow, high-intensity initial trigger spike (often index 10-20), followed by a broad envelope of high-frequency digitizer noise ("grass").
- Terminates abruptly at an electronic cutoff (index 640-850), returning to a flat baseline with rare, single-point dark counts.

#### Class 1
- Pure water ice spectrum consisting of a regular sequence of hydronium cluster peaks ($H_3O^+(H_2O)_n$) at mass 19, 37, 55, 73, 91... (index locations ~85, ~120, ~146, ~169, ~189).
- Early global maximum (index 80-100) followed by a rapid, step-like decay of peak heights towards the right.
- Valleys between peaks return completely to the flat baseline (no intermediate peak structures or elevated organic noise).
- Terminates in a sharp cutoff near index 650.

#### Class 2
- Organic-bearing or dirty water ice.
- Features the same hydronium cluster peak locations as Class 1, but has significant organic/saline contamination.
- Valleys between the major peaks are filled (elevated signal baseline) with unresolved organic background, secondary peaks, or high-frequency fluctuations.
- Global maximum is often in the index 180-300 range, showing massive, broad, unresolved molecular cluster sequences.
- Terminates in a sheer drop-off cliff around index 640-700.

#### Class 3
- Characterized by a continuous, highly elevated "mesa" plateau or unresolved complex organic mixture.
- Does not return to baseline between peaks throughout the spectrum (sustained signal above the noise floor).
- Typically features 3 to 4 broad, asymmetric "shark-fin" peak clusters with expanding periodicity (representing carbon clusters $C_n$ or heavy homologous organic series). Water peaks are absent or negligible.

#### Class 4
- Mineral/silicate-rich spectrum.
- Characterized by isolated, needle-sharp atomic spikes in the low-mass region (like $Mg^+$ at 24 Da, $Si^+$ at 28 Da, $Fe^+$ at 56 Da) with a very quiet baseline in between.
- Crucially, lacks the repeating, comb-like water cluster sequence ($H_3O^+(H_2O)_n$) of Class 1 and 2.
- Transitioning to broad mid-mass envelopes with a global maximum at index 280-350 and a distinct late cluster around 420-480, terminating in a hard cutoff near index 650.

#### Class 5
- Characterized by a bimodal extreme: an overwhelmingly intense primary peak in the early region (index 60-90, peaking near 70-75) with a long, trailing decay edge.
- The rest of the spectrum is a dense, uniform, low-amplitude "barcode" or "grass" band that crashes into a hard cliff at index 650.

#### Class 5-Na
- Bipartite structure dominated by Sodium chemistry.
- Erupts with a singular, overwhelmingly intense, sharp spike (the Sodium payload) at index 135-160.
- Followed by a delayed, prolonged, highly noisy, and completely unresolved plateau from index 200 to 650 (representing detector saturation, plasma shielding, or complex sodium-water clusters) that terminates in a hard cliff.

#### Class 3-P
- Characterized by a "burst-and-trail" signature consisting of broad, unresolved mass envelopes rather than sharp, isolated single-element lines.
- Features a massive primary complex (index ~200-220) which is the absolute maximum, followed by a distinct secondary peak around index ~330-345.
- Displays a unique elevated baseline/plateau past index 400 that never returns to zero (chemical noise plateau), with superimposed broad rhythmic hummocks, terminating in a rapid collapse/cutoff between index 650 and 750.

---

### CLASSIFICATION TASK:
Analyze the provided target spectrum image. Compare it to the reference examples and class profiles. Output your prediction using one of the following exact labels:
- "1"
- "2"
- "3"
- "4"
- "5"
- "5-Na"
- "3-P"
- "Noise"

Return the prediction strictly in the requested JSON format."""

BASELINE_SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT = """You are an expert Cosmic Dust Spectroscopist analyzing Cassini Cosmic Dust Analyzer (CDA) time-of-flight mass spectra.
Your task is to review and verify a first-pass prediction for a spectrum that was classified as either Class 3 (General Organics) or Class 3-P (Burst-and-Trail Agglomerate).

There is a common misclassification trap where VLMs misidentify a noisy or cluster-rich Class 3 baseline as a Class 3-P plateau. You must rigorously verify the prediction by evaluating these three criteria:

1. **Primary Peak Dominance:**
   - Class 3-P: Must have a single, explosive primary peak complex (index ~200-220) that is the absolute global maximum (y = 1.0) and dominates the rest of the spectrum by a factor of 5 to 10.
   - Class 3: Has multiple broad peak clusters (carbon chain clusters) of relatively comparable heights across the index range, without a single early maximum dominating the entire spectrum.

2. **Valley Depth between Clusters:**
   - Class 3: The valleys between the broad peak clusters drop significantly lower, returning close to the noise floor (down to y < 0.15).
   - Class 3-P: The signal remains highly elevated throughout.

3. **Baseline Plateau Cushion (past index 400):**
   - Class 3-P: Crucially, past index 400, the signal rests on a continuous, flat "chemical noise" plateau (baseline cushion) that remains highly elevated (y ≈ 0.3 to 0.5) and never returns to zero before dropping off sharply near index 650-750.
   - Class 3: The baseline past index 400 decays back close to zero.

You will be provided with:
- The target spectrum image.
- Reference examples of true Class 3 and Class 3-P spectra.

Analyze the image, critique the first-pass reasoning, and output your final prediction (either "3" or "3-P")."""

BASELINE_GENERAL_PROFILES = {
    "Noise": "Instrumental digitizer noise showing a narrow trigger spike and high-frequency 'grass', completely devoid of chemical peaks.",
    "1": "Pure water ice sequence of regularly spaced hydronium cluster peaks ($H_3O^+(H_2O)_n$) with an early global maximum and valleys returning fully to baseline.",
    "2": "Organic-bearing water ice featuring hydronium peaks similar to Class 1, but with prominent organic background noise filling the valleys between peaks.",
    "3": "General organic chains consisting of multiple broad, asymmetric 'shark-fin' peak clusters (carbon chains), lacking a singular early dominant peak.",
    "4": "Mineral/silicate spectrum showing isolated, needle-sharp atomic spikes (Mg+, Si+, Fe+) with a very quiet, flat baseline in between.",
    "5": "Bimodal extreme featuring an overwhelmingly intense primary peak in the early region (~70-75) with a long trailing decay and a uniform low-amplitude barcode band.",
    "5-Na": "Bipartite sodium chemistry dominated by a singular, intense early sodium spike (~135-160) followed by a delayed, noisy detector-saturation plateau.",
    "3-P": "Burst-and-trail signature showing an explosive early maximum (~200), a distinct secondary peak (~340), and a continuous elevated chemical noise baseline past index 400."
}

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def extract_text_from_papers(papers_dir="papers"):
    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
    print(f"Extracting scientific content from {len(pdf_files)} papers...")
    paper_texts = {}
    for pdf in sorted(pdf_files):
        path = os.path.join(papers_dir, pdf)
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        paper_texts[pdf] = text
        print(f"  - Read {pdf}: {len(reader.pages)} pages, {len(text)} chars")
    return paper_texts

def get_genai_client():
    return genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

def update_cda_dust_agent_prompts(refined_system_instruction):
    prompts_file = "cda_dust_agent/prompts.py"
    if not os.path.exists(prompts_file):
        print(f"Warning: {prompts_file} not found.")
        return
        
    with open(prompts_file, "r") as f:
        content = f.read()

    new_content = f'SYSTEM_INSTRUCTION_TEXT = """{refined_system_instruction}"""\n\n'
    if "ANNOTATION_USER_PROMPT =" in content:
        remaining = content.split("ANNOTATION_USER_PROMPT =", 1)[1]
        new_content += "ANNOTATION_USER_PROMPT =" + remaining
    else:
        new_content += content

    with open(prompts_file, "w") as f:
        f.write(new_content)
    print(f"Updated {prompts_file} with refined system instruction from Agent 3 (Referee).")

# ---------------------------------------------------------
# Agent 1: Academic Research Agent
# ---------------------------------------------------------
class AcademicResearchAgent:
    def __init__(self, client):
        self.client = client

    def run(self, paper_texts, baseline_system_instruction, baseline_self_correct_instruction, baseline_general_profiles):
        print("\n=== Agent 1: Academic Research Agent Starting ===")
        
        paper_summaries = ""
        for pdf_name, text in paper_texts.items():
            paper_summaries += f"\n--- PAPER: {pdf_name} ---\n{text[:6000]}\n[...truncated for context window...]\n"

        prompt = f"""You are the Academic Research Agent specializing in Cassini Cosmic Dust Analyzer (CDA) Time-of-Flight Mass Spectrometry.

TASK:
You are provided with key scientific papers on Cassini CDA dust mass spectra and the baseline prompt instructions used for VLM classification.
Your goal is to critically analyze the scientific literature and retrieve key physical, chemical, and spectral information that will improve the classification results and refine the prompt instructions (especially for distinguishing Class 3 vs Class 3-P, as well as Class 1, 2, 4, 5, 5-Na, Noise).

BASELINE SYSTEM INSTRUCTION (STAGE 1):
{baseline_system_instruction}

BASELINE SYSTEM INSTRUCTION (STAGE 2 SELF-CORRECT):
{baseline_self_correct_instruction}

BASELINE GENERAL PROFILES:
{json.dumps(baseline_general_profiles, indent=2)}

LITERATURE EXTRACTS:
{paper_summaries}

Please output a comprehensive ACADEMIC RESEARCH PROPOSAL REPORT containing:
1. Key scientific insights from the papers regarding CDA mass resolution, time-of-flight shifting/stretching physics, chemical ion species (hydronium clusters, refractory silicates, macro-organics, sodium payloads, burst-and-trail impact dynamics).
2. Specific critical criteria for differentiating Class 3 (General Organics) vs Class 3-P (Burst-and-Trail Agglomerate) based on physical impact mechanisms and mass spectrum characteristics.
3. Concrete recommendations to update `SYSTEM_INSTRUCTION_TEXT`, `SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT`, and `GENERAL_PROFILES` to maximize classification accuracy and F1 score.
"""
        response = safe_generate_content(self.client, MODEL_ID, prompt)
        proposal = response.text.strip()
        print("Academic Research Agent Proposal generated.")
        return proposal

# ---------------------------------------------------------
# Agent 2: Academic Critic Agent
# ---------------------------------------------------------
class AcademicCriticAgent:
    def __init__(self, client):
        self.client = client

    def run(self, research_proposal, baseline_system_instruction):
        print("\n=== Agent 2: Academic Critic Agent Starting ===")
        
        prompt = f"""You are the Academic Critic Agent. Your role is to critically review and challenge the recommendations made by the Academic Research Agent for the Cassini CDA classification prompt system.

TASK:
Evaluate the proposed prompt modifications and scientific arguments from the Academic Research Agent.
Identify potential weaknesses, risks of overfitting, physical inconsistencies with CDA instrument physics, ambiguous wording, or edge cases where the proposed rules might cause false positives (e.g., misclassifying Class 2 dirty water ice as Class 3 organics, or misclassifying Class 3-P as Class 5-Na).

RESEARCH AGENT PROPOSAL:
{research_proposal}

BASELINE SYSTEM INSTRUCTION:
{baseline_system_instruction}

Please provide a detailed ACADEMIC CRITIQUE REPORT:
1. Points of Agreement: Recommendations that are scientifically sound, clear, and beneficial for VLM reasoning.
2. Points of Contention & Counterarguments: Specific proposed rules that are overly restrictive, prone to false positives, or physically misinformed.
3. Refinements & Alternative Formulations: Specific modifications to sharpen the distinction between Class 3 and Class 3-P, and between Class 1/2/4/5/5-Na.
"""
        response = safe_generate_content(self.client, MODEL_ID, prompt)
        critique = response.text.strip()
        print("Academic Critic Agent Report generated.")
        return critique

# ---------------------------------------------------------
# Agent 3: Academic Referee Agent
# ---------------------------------------------------------
class AcademicRefereeAgent:
    def __init__(self, client):
        self.client = client

    def run(self, research_proposal, critic_report):
        print("\n=== Agent 3: Academic Referee Agent Starting ===")
        
        prompt = f"""You are the Academic Referee Agent (Editor-in-Chief & Final Decision Maker).
You have reviewed both the Academic Research Agent's proposal and the Academic Critic Agent's critique regarding the Cassini CDA dust mass spectra classification prompts.

TASK:
Weigh both sides carefully and make final authoritative decisions on the prompt enhancements.
Synthesize the final, production-ready prompt structures for the CDA VLM Classifier.

RESEARCH PROPOSAL:
{research_proposal}

CRITIC REPORT:
{critic_report}

You MUST produce your response in TWO SECTIONS:

SECTION 1: REFEREE DECISION SUMMARY (Markdown text explaining your decisions, rationale, and final guidelines).

SECTION 2: PROMPT DEFINITIONS JSON
Provide a valid JSON block enclosed in ```json ... ``` with the exact updated string variables:
{{
  "SYSTEM_INSTRUCTION_TEXT": "<refined Stage 1 system instruction text>",
  "SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT": "<refined Stage 2 self-correction system instruction text>",
  "GENERAL_PROFILES": {{
    "Noise": "<refined description>",
    "1": "<refined description>",
    "2": "<refined description>",
    "3": "<refined description>",
    "4": "<refined description>",
    "5": "<refined description>",
    "5-Na": "<refined description>",
    "3-P": "<refined description>"
  }}
}}
"""
        response = safe_generate_content(self.client, MODEL_ID, prompt)
        full_text = response.text.strip()
        
        decision_text = full_text
        json_obj = None
        if "```json" in full_text:
            parts = full_text.split("```json")
            decision_text = parts[0].strip()
            json_str = parts[1].split("```")[0].strip()
            try:
                json_obj = json.loads(json_str)
            except Exception as e:
                print(f"Warning: Failed to parse referee JSON block: {e}")
        
        print("Academic Referee Agent Decisions synthesized.")
        return decision_text, json_obj, full_text

# ---------------------------------------------------------
# Agent 4: General CDA Dust Agent Pipeline Execution
# ---------------------------------------------------------
class Agent4CdaWorkflowExecution:
    def __init__(self):
        pass

    async def run(self):
        print("\n=== Agent 4: Running General CDA Dust Agent Pipeline ===")
        from cda_dust_agent.config import Config
        from cda_dust_agent.agent import root_agent
        from google.adk.sessions.in_memory_session_service import InMemorySessionService
        from google.adk.agents.invocation_context import InvocationContext

        configs = Config()
        configs.agent_settings.inference_path = "batch"
        configs.agent_settings.location = "global"
        configs.agent_settings.project_id = PROJECT_ID
        configs.agent_settings.bucket_name = BUCKET_NAME
        configs.agent_settings.fetch_data = True
        configs.agent_settings.test_mode = False

        session_service = InMemorySessionService()
        session = await session_service.create_session(app_name="cda_app", user_id="user_1", session_id="multi_agent_session")
        session.state["fsa_state"] = "done"
        session.state["few_shot_examples"] = []
        session.state["used_ids"] = []

        ctx = InvocationContext(
            session=session,
            session_service=session_service,
            invocation_id="cda_multi_agent_pipeline_invocation",
            agent=root_agent
        )

        async for event in root_agent._run_async_impl(ctx):
            if event.content and event.content.parts:
                print(f"[{event.author}] {event.content.parts[0].text}")

        print("\n=================================================================")
        print(" GENERAL CDA DUST AGENT PIPELINE EXECUTED SUCCESSFULLY ")
        print("=================================================================")

# ---------------------------------------------------------
# Orchestrator Workflow
# ---------------------------------------------------------
def main():
    print("=================================================================")
    print(" CASSINI CDA MULTI-AGENT ARCHITECTURE WORKFLOW ")
    print("=================================================================")

    client = get_genai_client()

    # 1. Read paper PDFs
    paper_texts = extract_text_from_papers("papers")

    # 2. Agent 1: Academic Research Agent
    agent1 = AcademicResearchAgent(client)
    proposal = agent1.run(paper_texts, BASELINE_SYSTEM_INSTRUCTION_TEXT, BASELINE_SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT, BASELINE_GENERAL_PROFILES)
    with open(os.path.join(OUTPUT_DIR, "academic_research_proposal.md"), "w") as f:
        f.write(proposal)

    # 3. Agent 2: Academic Critic Agent
    agent2 = AcademicCriticAgent(client)
    critique = agent2.run(proposal, BASELINE_SYSTEM_INSTRUCTION_TEXT)
    with open(os.path.join(OUTPUT_DIR, "academic_critic_report.md"), "w") as f:
        f.write(critique)

    # 4. Agent 3: Academic Referee Agent
    agent3 = AcademicRefereeAgent(client)
    decision_text, json_obj, full_referee_response = agent3.run(proposal, critique)
    with open(os.path.join(OUTPUT_DIR, "academic_referee_decision.md"), "w") as f:
        f.write(full_referee_response)

    system_instruction = BASELINE_SYSTEM_INSTRUCTION_TEXT
    self_correct_instruction = BASELINE_SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT
    general_profiles = BASELINE_GENERAL_PROFILES

    if json_obj:
        system_instruction = json_obj.get("SYSTEM_INSTRUCTION_TEXT", system_instruction)
        self_correct_instruction = json_obj.get("SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT", self_correct_instruction)
        general_profiles = json_obj.get("GENERAL_PROFILES", general_profiles)
        print("Successfully updated prompt definitions from Referee JSON decision.")
    else:
        print("Referee JSON parsing did not return structured object, using default baseline prompts.")

    # Update cda_dust_agent/prompts.py with the refined prompt coming from Agent 3
    update_cda_dust_agent_prompts(system_instruction)

    with open(os.path.join(OUTPUT_DIR, "refined_prompts.py"), "w") as f:
        f.write(f'REFINED_SYSTEM_INSTRUCTION_TEXT = """{system_instruction}"""\n\n')
        f.write(f'REFINED_SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT = """{self_correct_instruction}"""\n\n')
        f.write(f'REFINED_GENERAL_PROFILES = {json.dumps(general_profiles, indent=2)}\n')

    # 5. Agent 4: Execute general CDA_dust_agent pipeline
    agent4 = Agent4CdaWorkflowExecution()
    asyncio.run(agent4.run())

    print("\n=================================================================")
    print(" MULTI-AGENT WORKFLOW COMPLETED SUCCESSFULLY ")
    print("=================================================================")

if __name__ == "__main__":
    main()
