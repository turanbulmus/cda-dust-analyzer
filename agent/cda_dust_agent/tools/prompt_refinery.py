import os
import json
import base64
import numpy as np
import pandas as pd
import asyncio
from google import genai
from google.genai import types
from ..config import Config
from .utils import generate_spectrum_image_bytes
import logging
import subprocess
import time
from datetime import datetime
from google.cloud import storage
import vertexai
from vertexai import generative_models
from vertexai import types as vx_types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

configs = Config()

async def get_spectrum_description(client: genai.Client, spectrum: np.ndarray, label: str, sclk: str) -> str:
    """Generates a detailed description of the spectrum using Gemini (Local)."""
    model_id = configs.agent_settings.model
    img_bytes = generate_spectrum_image_bytes(spectrum, title=f"{label} Sample {sclk}")
    
    prompt = f"Analyze this Time-of-Flight mass spectrum for a dust particle of Class '{label}'. Describe the key patterns, peak sequences, baseline characteristics, and any unique features observed. Be detailed and specific."
    
    contents = [
        types.Content(role="user", parts=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        ])
    ]
    
    try:
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=contents,
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Failed to generate description for {sclk}: {e}")
        return f"Failed to generate description: {e}"

async def run_prompt_refinery(autonomous: bool = True, max_levels: int = 5, use_batch: bool = False, project_id: str = "turan-genai-bb") -> str:
    """Runs the hierarchical prompt refinery study.
    
    If autonomous=True, the agent decides how many patterns to produce at each level
    and when to stop consolidating.
    """
    logger.info(f"Starting prompt refinery study (Autonomous={autonomous})...")
    
    train_file = "cda_dust_agent/data/raw/cda_train.parquet"
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
        
    df = pd.read_parquet(train_file)
    
    # Stratified sampling of 30 observations per class
    classes = df['class'].unique()
    sampled_dfs = []
    for cls in classes:
        cls_df = df[df['class'] == cls]
        n_samples = min(30, len(cls_df))
        if n_samples > 0:
            sampled_dfs.append(cls_df.sample(n=n_samples, random_state=42))
            
    sample_df = pd.concat(sampled_dfs).reset_index(drop=True)
    logger.info(f"Sampled {len(sample_df)} observations across {len(classes)} classes.")
    
    # Save the sample data
    os.makedirs("cda_dust_agent/data/prompt_optimizer", exist_ok=True)
    sample_df.to_parquet("cda_dust_agent/data/prompt_optimizer/sampled_train.parquet")
    
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ["GOOGLE_API_USE_MTLS"] = "never"
    client = genai.Client(vertexai=True, project=project_id, location="global")
    bucket_name = configs.agent_settings.bucket_name.replace("gs://", "")
    storage_client = storage.Client(project=project_id)
    
    # Initialize Vertex AI client for prompt management
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        vx_client = vertexai.Client(project=project_id, location="us-central1")
        
        before_prompt_text = "Analyze this Time-of-Flight mass spectrum for a dust particle of Class '{label}'. Describe the key patterns, peak sequences, baseline characteristics, and any unique features observed. Be detailed and specific."
        
        before_prompt = vx_client.prompts.create(
            prompt_id=f"cda_desc_template_{timestamp}",
            prompt=vx_types.Prompt(
                prompt_data=vx_types.PromptData(
                    contents=[
                        generative_models.Content(role="user", parts=[
                            generative_models.Part.from_text(before_prompt_text)
                        ])
                    ],
                    model=configs.agent_settings.model
                )
            )
        )
        logger.info(f"Saved 'before' prompt to management tool: {before_prompt.name}")
    except Exception as e:
        logger.error(f"Failed to save 'before' prompt: {e}")
        
    # Level 1: Generate descriptions for each sample
    logger.info("Level 1: Generating descriptions for each sample...")
    
    current_results = []
    
    if not use_batch:
        semaphore = asyncio.Semaphore(5)
        
        async def process_row(row):
            async with semaphore:
                desc = await get_spectrum_description(client, np.array(row['spectrum']), row['class'], str(row['sclk']))
                return {
                    "sclk": row['sclk'],
                    "class": row['class'],
                    "description": desc
                }
                
        tasks = [process_row(row) for _, row in sample_df.iterrows()]
        level_1_results = await asyncio.gather(*tasks)
        current_results = level_1_results
    else:
        # Batch Mode for Level 1 (Simulated skeleton as before)
        logger.info("Batch L1 would run here.")
        # We assume current_results is populated from the batch output.
        
    # Save Level 1 results
    os.makedirs("cda_dust_agent/data/results", exist_ok=True)
    pd.DataFrame(current_results).to_csv("cda_dust_agent/data/results/prompt_study_level_1.csv", index=False)
    
    # Initialize descriptions for next levels
    current_descriptions = {} # class -> list of descriptions
    for res in current_results:
        cls = res['class']
        desc = res['description']
        if cls not in current_descriptions:
            current_descriptions[cls] = []
        current_descriptions[cls].append(desc)
        
    level_num = 2
    active_classes = list(current_descriptions.keys())
    
    final_patterns = {} # class -> list of final patterns
    
    while active_classes and level_num <= max_levels:
        logger.info(f"Level {level_num}: Consolidating patterns for {len(active_classes)} classes...")
        
        next_descriptions = {}
        new_active_classes = []
        
        if not use_batch:
            async def process_class_autonomous(cls, descs):
                logger.info(f"Analyzing {len(descs)} patterns for Class {cls}...")
                
                prompt = (
                    f"Here are the descriptions of spectra for Class '{cls}'.\n"
                    f"1. Identify how many distinct characteristic patterns are present in these descriptions.\n"
                    f"2. Provide a detailed description for each distinct pattern.\n"
                    f"3. Decide if these patterns can be consolidated further in a next step to reach a single unified pattern for this class, "
                    f"or if they represent the fundamental distinct types for this class that should not be merged.\n"
                    f"Return the result as a JSON object."
                )
                
                text_parts = [prompt]
                for i, d in enumerate(descs):
                    text_parts.append(f"Description {i+1}:\n{d}\n")
                    
                prompt_text = "\n".join(text_parts)
                
                try:
                    response = await client.aio.models.generate_content(
                        model=configs.agent_settings.model,
                        contents=prompt_text,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    "patterns": types.Schema(
                                        type=types.Type.ARRAY,
                                        items=types.Schema(type=types.Type.STRING),
                                        description="List of identified distinct patterns."
                                    ),
                                    "can_consolidate_further": types.Schema(
                                        type=types.Type.BOOLEAN,
                                        description="True if these patterns can be consolidated further in a next level."
                                    ),
                                    "explanation": types.Schema(
                                        type=types.Type.STRING,
                                        description="Explanation for the decision."
                                    )
                                },
                                required=["patterns", "can_consolidate_further", "explanation"]
                            )
                        )
                    )
                    result_data = json.loads(response.text)
                    return cls, result_data
                except Exception as e:
                    logger.error(f"Failed at Level {level_num} for Class {cls}: {e}")
                    return cls, {"patterns": [f"Failed to consolidate: {e}"], "can_consolidate_further": False, "explanation": str(e)}
                    
            tasks = [process_class_autonomous(cls, current_descriptions[cls]) for cls in active_classes]
            level_results = await asyncio.gather(*tasks)
            
            for cls, result_data in level_results:
                patterns = result_data.get("patterns", [])
                can_consolidate = result_data.get("can_consolidate_further", False)
                
                logger.info(f"Class {cls}: Identified {len(patterns)} patterns. Can consolidate further: {can_consolidate}")
                
                if can_consolidate and len(patterns) > 1:
                    next_descriptions[cls] = patterns
                    new_active_classes.append(cls)
                else:
                    # We stop here for this class
                    final_patterns[cls] = patterns
                    
        else:
            # Batch Mode for Autonomous Level
            logger.info(f"Level {level_num}: Batch processing for {len(active_classes)} classes...")
            jsonl_file = f"cda_dust_agent/data/prompt_optimizer/prompt_study_requests_l{level_num}.jsonl"
            
            requests = []
            for cls in active_classes:
                descs = current_descriptions[cls]
                prompt = (
                    f"Here are the descriptions of spectra for Class '{cls}'.\n"
                    f"1. Identify how many distinct characteristic patterns are present in these descriptions.\n"
                    f"2. Provide a detailed description for each distinct pattern.\n"
                    f"3. Decide if these patterns can be consolidated further in a next step to reach a single unified pattern for this class, "
                    f"or if they represent the fundamental distinct types for this class that should not be merged.\n"
                    f"Return the result as a JSON object."
                )
                
                text_parts = [prompt]
                for i, d in enumerate(descs):
                    text_parts.append(f"Description {i+1}:\n{d}\n")
                    
                prompt_text = "\n".join(text_parts)
                
                request = {
                    "request": {
                        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "responseSchema": {
                                "type": "OBJECT",
                                "properties": {
                                    "patterns": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "can_consolidate_further": {"type": "BOOLEAN"},
                                    "explanation": {"type": "STRING"}
                                },
                                "required": ["patterns", "can_consolidate_further", "explanation"]
                            }
                        }
                    }
                }
                requests.append(json.dumps(request))
                
            with open(jsonl_file, 'w') as f:
                for req in requests:
                    f.write(req + '\n')
                    
            # Upload, Submit, Poll, Download (Simulated here)
            logger.info(f"Batch L{level_num} submitted for {len(active_classes)} classes (simulated).")
            # We would parse results and populate next_descriptions and new_active_classes
            # For now, assume we switch to final if simulation is not complete or mock it.
            # To make it work in this draft, let's assume it stops or we use local fallback if not implemented fully.
            
        current_descriptions = next_descriptions
        active_classes = new_active_classes
        
        # Save intermediate results
        with open(f"cda_dust_agent/data/results/prompt_study_level_{level_num}.json", "w") as f:
            json.dump(final_patterns, f, indent=2) # Save what we have finalized so far
            
        level_num += 1
        
    # Ensure all classes have final patterns (if they stopped early)
    for cls in current_descriptions:
        if cls not in final_patterns:
            final_patterns[cls] = current_descriptions[cls]
            
    # Generate Final Prompt
    logger.info("Generating final refined system prompt...")
    system_prompt = "You are an expert system for classifying Cassini Cosmic Dust Analyzer (CDA) Time-of-Flight mass spectra.\n"
    system_prompt += "Here are the characteristic patterns for each class identified through autonomous hierarchical analysis:\n\n"
    
    for cls, patterns in final_patterns.items():
        system_prompt += f"### Class {cls}\n"
        for i, pattern in enumerate(patterns):
            if len(patterns) > 1:
                system_prompt += f"Pattern {i+1}:\n{pattern}\n\n"
            else:
                system_prompt += f"{pattern}\n\n"
                
    system_prompt += "Use these patterns to classify the provided spectrum. Return the classification in the requested JSON format."
    
    with open("cda_dust_agent/refined_prompt.txt", "w") as f:
        f.write(system_prompt)
        
    # Generate Documentation
    logger.info("Generating documentation...")
    doc = f"# Prompt Improvement Study Documentation\n\n"
    doc += f"This study aimed to improve the prompts for each class using the training file.\n"
    doc += f"We used an **autonomous hierarchical approach** where the agent decided the levels and consolidation.\n\n"
    doc += f"## Final Patterns Identified\n"
    for cls, patterns in final_patterns.items():
        doc += f"### Class {cls}\n"
        doc += f"Identified {len(patterns)} fundamental patterns.\n\n"
        
    doc += f"## Final Prompt\n"
    doc += f"The resulting system prompt is saved to `refined_prompt.txt`.\n"
    
    with open("cda_dust_agent/prompt_study_documentation.md", "w") as f:
        f.write(doc)
        
    # Save prompt after optimization
    try:
        after_prompt = vx_client.prompts.create(
            prompt_id=f"cda_refined_prompt_{timestamp}",
            prompt=vx_types.Prompt(
                prompt_data=vx_types.PromptData(
                    contents=[
                        generative_models.Content(role="user", parts=[
                            generative_models.Part(text=system_prompt)
                        ])
                    ],
                    model=configs.agent_settings.model
                )
            )
        )
        logger.info(f"Saved 'after' prompt to management tool: {after_prompt.name}")
    except Exception as e:
        logger.error(f"Failed to save 'after' prompt: {e}")
        
    return system_prompt
