import os
import io
import json
import base64
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv('.env')

PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'turan-genai-bb')
MODEL_ID = 'gemini-2.5-flash'
OUTPUT_DIR = "study_results_multi_agent"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of 3-P false positives from the 2000 evaluation run
SAMPLES = [
    # True Class 2
    {"sclk": "1557518477", "true_class": "2", "explanation": "The spectrum displays an explosive multi-peak primary complex reaching a global maximum of near 1.0 around index 170, a distinct secondary cluster near index 320, and a continuously elevated chemical noise baseline cushion past index 400 that stays well above zero."},
    {"sclk": "1589069030", "true_class": "2", "explanation": "The spectrum displays an explosive primary complex reaching y = 1.0 around index 200, followed by a secondary peak around index 290 and a highly elevated chemical noise baseline plateau (~0.5) that remains elevated across the remaining mass range."},
    {"sclk": "1506226335", "true_class": "2", "explanation": "The spectrum displays an explosive primary complex reaching y = 1.0 around index 200, followed by distinct secondary envelopes, and rests on a continuously elevated, flat chemical noise baseline cushion of y ≈ 0.45 past index 350."},
    {"sclk": "1498483622", "true_class": "2", "explanation": "The spectrum displays an explosive primary multi-peak complex near index 280, followed by a secondary envelope peak around index 350, all resting on an elevated, continuous chemical noise baseline cushion that remains well above zero."},
    {"sclk": "1543799319", "true_class": "2", "explanation": "The spectrum displays a typical porous impact burst-and-trail signature, showing an explosive primary multi-peak cluster complex peaking near y=1.0, followed by a secondary peak and a continuous elevated baseline cushion that remains elevated above 0.5 until the end of the window."},
    
    # True Class 3
    {"sclk": "1519585197", "true_class": "3", "explanation": "The spectrum displays an explosive, unresolved multi-peak primary complex reaching a global maximum of 1.0, followed by a prominent secondary envelope near index 370 and a flat, elevated chemical noise baseline cushion that remains around y ≈ 0.6 past index 400."},
    {"sclk": "1557519216", "true_class": "3", "explanation": "The spectrum displays a characteristic burst-and-trail profile with an explosive multi-peak primary complex near index 200, a prominent secondary peak around index 345, and a continuously elevated chemical noise baseline cushion (y ≈ 0.6 to 0.8) past index 400."},
    {"sclk": "1543799509", "true_class": "3", "explanation": "The spectrum displays an explosive primary multi-peak complex near index 200 that constitutes the global maximum, followed by a wide secondary envelope and an elevated chemical noise baseline cushion that remains high past index 400."},
    {"sclk": "1506250096", "true_class": "3", "explanation": "The spectrum displays an explosive primary multi-peak cluster complex peaking at y = 1.0, followed by a distinct secondary organic envelope around index 360, and is built upon a continuously elevated, flat chemical noise baseline cushion (y ≈ 0.45) that stays elevated throughout."},
    {"sclk": "1557519439", "true_class": "3", "explanation": "The spectrum displays a porous particle impact profile characterized by a prominent primary complex reaching y = 1.0, followed by a secondary peak and a continuous chemical noise baseline cushion that stays elevated around y ≈ 0.40 before dropping towards the end of the scan."},
    
    # True Class Noise
    {"sclk": "1533132722", "true_class": "Noise", "explanation": "The spectrum displays an extremely broad, unresolved primary multi-peak complex that peaks near index 180 and slowly decays to an elevated, continuous chemical noise baseline cushion around y ≈ 0.55. It matches the Porous/Agglomerate (Class 3-P) profile characterized by this specific topography and a failure to drop back to near-zero levels."},
    {"sclk": "1489024612", "true_class": "Noise", "explanation": "The spectrum displays a broad, explosive primary complex peaking at 1.0 around index 200, followed by a flat elevated chemical noise baseline cushion around 0.3, and a distinct secondary rising envelope toward the end."},
    
    # True Class 5
    {"sclk": "1506298659", "true_class": "5", "explanation": "The spectrum displays an explosive, unresolved multi-peak primary complex peaking near y = 1.0, followed by a distinct secondary peak structure and a continuous elevated chemical noise baseline cushion that rests around y = 0.25 to 0.35 before gradually tapering off."},
    {"sclk": "1589068770", "true_class": "5", "explanation": "The spectrum displays a burst-and-trail profile featuring an explosive primary multi-peak complex near index 175 and a secondary shoulder-like peak near index 245. Past these primary features, the spectrum settles into a flat, elevated chemical noise baseline cushion around y ≈ 0.38 that persists across the remainder of the spectrum, confirming a porous agglomerate impact signature."}
]

def get_genai_client():
    return genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

def generate_spectrum_plot(spectrum_data, title):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
    ax.plot(spectrum_data, color='black', linewidth=1.5)
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 630)
    ax.grid(True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

async def run_study():
    print("Loading test dataset to retrieve spectrum arrays...")
    df = pd.read_parquet("cda_dust_agent/data/testing/cda_test.parquet")
    df['sclk'] = df['sclk'].astype(str)
    
    client = get_genai_client()
    discussions = []
    
    for idx, sample in enumerate(SAMPLES, 1):
        sclk = sample['sclk']
        true_cls = sample['true_class']
        explanation = sample['explanation']
        
        # Get spectrum data
        row = df[df['sclk'] == sclk]
        if row.empty:
            print(f"Warning: SCLK {sclk} not found in cda_test.parquet. Skipping.")
            continue
            
        spectrum = row.iloc[0]['spectrum']
        qi = row.iloc[0]['qi_ampl']
        
        # Generate plot image
        img_bytes = generate_spectrum_plot(spectrum, f"SCLK: {sclk} (True Class: {true_cls})")
        
        print(f"\n[{idx}/{len(SAMPLES)}] Analyzing SCLK {sclk} (True Class: {true_cls})...")
        
        # 1. Academic Research Agent
        academic_prompt = f"""You are the Academic Research Agent specializing in Cassini Cosmic Dust Analyzer (CDA) Time-of-Flight mass spectra.
This target spectrum (SCLK: {sclk}, QI_AMPL: {qi:.3e} C) was misclassified as Class 3-P (Burst-and-Trail Agglomerate), but its true ground class is Class {true_cls}.
The model's incorrect explanation was:
"{explanation}"

Please review the target spectrum image and perform a detailed scientific analysis:
1. Identify the visual/physical features that caused the model to misclassify this as Class 3-P (e.g. baseline shape, peak structures, valleys, trigger artifacts, digitizer grass).
2. Contrast this spectrum with a true Class 3-P spectrum. How does this spectrum differ from the actual physical characteristics of Class 3-P (which requires an explosive primary impact complex at index 200, a secondary peak at index 330-345, and a flat, continuous chemical noise baseline cushion of y ≈ 0.25-0.50 past index 400)?
3. Provide concrete recommendations to improve our prompts, classification rules, or plot visualizations (such as zooming into specific index regions, plotting annotations of detected peaks, or denoising the signal) to ensure it is correctly classified as Class {true_cls} instead of 3-P.
"""
        academic_contents = [
            types.Content(role="user", parts=[
                types.Part.from_text(text=academic_prompt),
                types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            ])
        ]
        
        academic_response = client.models.generate_content(
            model=MODEL_ID,
            contents=academic_contents,
            config=types.GenerateContentConfig(
                system_instruction="You are an expert Cosmic Dust Spectroscopist providing scientific feedback on mass spectrum classification failures."
            )
        )
        academic_analysis = academic_response.text.strip()
        print("  - Academic Agent analysis generated.")
        
        # 2. Academic Critic Agent
        critic_prompt = f"""You are the Academic Critic Agent. Your role is to critically review and challenge the findings and recommendations made by the Academic Research Agent.
We are analyzing SCLK {sclk} (True Class: {true_cls}, Predicted Class: 3-P).

ACADEMIC RESEARCH AGENT'S ANALYSIS:
{academic_analysis}

Please evaluate the proposal:
1. Identify potential scientific flaws or weaknesses in the Academic Agent's reasoning.
2. Critique their recommendations (prompt updates or tool changes). Are they practical? Do they risk overfitting to this specific spectrum? Would implementing their changes cause false positives for other classes (e.g. misclassifying true Class 3-P as Class {true_cls})?
3. Provide a consolidated critique and suggest alternative refinements if needed.
"""
        critic_contents = [
            types.Content(role="user", parts=[
                types.Part.from_text(text=critic_prompt),
                types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            ])
        ]
        
        critic_response = client.models.generate_content(
            model=MODEL_ID,
            contents=critic_contents,
            config=types.GenerateContentConfig(
                system_instruction="You are a machine learning engineer and spectroscopic critic reviewing classification prompt improvements for cosmic dust spectra."
            )
        )
        critic_analysis = critic_response.text.strip()
        print("  - Academic Critic Agent analysis generated.")
        
        discussions.append({
            "sclk": sclk,
            "true_class": true_cls,
            "qi_ampl": float(qi),
            "explanation": explanation,
            "academic_analysis": academic_analysis,
            "critic_analysis": critic_analysis
        })
        
    # Save discussions to JSON
    json_path = os.path.join(OUTPUT_DIR, "academic_critic_3p_discussions.json")
    with open(json_path, 'w') as f:
        json.dump(discussions, f, indent=2)
    print(f"\nSaved all discussions to {json_path}")
    
    # Generate final consolidated MD report
    await generate_md_report(discussions)

async def generate_md_report(discussions):
    client = get_genai_client()
    
    summary_prompt = """You are the Academic Referee Agent (Editor-in-Chief).
You have been provided with a series of Academic vs. Critic discussions regarding Cassini CDA spectra that were misclassified as Class 3-P (Burst-and-Trail Agglomerate) when they belonged to other classes (Noise, Class 2, Class 3, Class 5).

Here is the JSON containing the discussions for each sample:
{discussions_json}

Your task is to compile a comprehensive, publication-grade academic report titled:
# CASSINI CDA DUST ANALYZER: DEEP FAILURES ANALYSIS STUDY OF CLASS 3-P (BURST-AND-TRAIL) FALSE POSITIVES

The report MUST include:
1. Executive Summary: Overview of the 3-P classification trap and the key statistics of the false positives.
2. Detailed Analysis per True Class (Noise, Class 2, Class 3, Class 5) that were misclassified as 3-P, summarizing the key visual/physical triggers of the misclassifications and the consensus explanations from the dialogue.
3. Systemic Recommendations for:
   - Prompt Enhancements (updating the system instructions/class profiles to prevent the 3-P trap).
   - Tool/Visualization Improvements (e.g. peak annotation, zoom boxes, baseline calculation, and other feature-engineering tools that can be plotted or supplied as metadata to the VLM).
4. A Synthesized Set of Prompt Guidelines (specifically refining the Class 3-P definition and its exclusion criteria).

Output the report in clean markdown format. Do not use code blocks for the markdown text itself, but you can use code blocks for prompt templates or python code suggestions.
"""
    
    discussions_str = json.dumps(discussions, indent=2)
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=summary_prompt.format(discussions_json=discussions_str),
        config=types.GenerateContentConfig(
            system_instruction="You are the Editor-in-Chief synthesizing scientific findings and formulating prompt engineering policies."
        )
    )
    
    report_content = response.text.strip()
    report_path = os.path.join(OUTPUT_DIR, "academic_critic_3p_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Generated final report at {report_path}")

if __name__ == "__main__":
    asyncio.run(run_study())
