import os
import json
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

from ..config import Config

from ..tools.utils import get_few_shot_candidates, generate_spectrum_image_bytes
from ..prompts import ANNOTATION_USER_PROMPT, SYSTEM_INSTRUCTION_TEXT

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))

configs = Config()

class FewShotAnnotationAgent(BaseAgent):
    """Automatically generates few-shot explanations using Gemini."""
    data_path: str = "cda_dust_agent/data/raw/cda_train.parquet"
    
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        import os
        import json
        if configs.agent_settings.inference_path not in ["local", "batch"]:
            return
            
        # If session state already has the loaded examples, we can return early
        if ctx.session.state.get("fsa_state") == "done" and "few_shot_examples" in ctx.session.state:
            return
            
        fsa_state = "auto_annotate"
        
        # Initialize
        os.makedirs("cda_dust_agent/data/input/examples", exist_ok=True)
        cache_file = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
        
        use_cache = not configs.agent_settings.force_new_annotations
        
        if use_cache and os.path.exists(cache_file):
            import json
            import base64
            # Verify cache has items
            with open(cache_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    cached_ex = []
                    for line in lines:
                        if line.strip():
                            entry = json.loads(line)
                            if "image_base64" in entry:
                                entry["image"] = base64.b64decode(entry["image_base64"])
                            cached_ex.append(entry)
                    
                    ctx.session.state["few_shot_examples"] = cached_ex
                    ctx.session.state["used_ids"] = [ex["sclk"] for ex in cached_ex]
                    ctx.session.state["fsa_state"] = "done"
                    yield log_and_yield(self.name, f"Loaded {len(cached_ex)} cached examples. Proceeding to inference...")
                    return
        
        # Clear the cache file since we are intentionally ignoring it or it's empty
        if not use_cache and os.path.exists(cache_file):
            os.remove(cache_file)
        
        yield log_and_yield(self.name, "Starting automatic few-shot annotation with Gemini...")
            
        if fsa_state == "auto_annotate":
            n_per_class = configs.agent_settings.few_shot_n
            df = pd.read_parquet(self.data_path)
            candidates = get_few_shot_candidates(df, n_per_class=n_per_class)
            
            if not candidates:
                 ctx.session.state["fsa_state"] = "done"
                 return
                 
            ctx.session.state["few_shot_examples"] = []
            
            from google import genai
            from google.genai import types
            import base64
            import numpy as np
            
            client = genai.Client(vertexai=True, project=configs.agent_settings.project_id, location="us-central1")
            model_id = configs.agent_settings.model
            
            yield log_and_yield(self.name, f"Generating explanations for {len(candidates)} examples ({n_per_class} per class) using {model_id}...")
            
            os.makedirs("cda_dust_agent/data/input/examples", exist_ok=True)
            os.makedirs("cda_dust_agent/data/annotated_spectra", exist_ok=True)
            os.makedirs("cda_dust_agent/data/input/store", exist_ok=True)
            cache_file = "cda_dust_agent/data/input/examples/cached_examples.jsonl"
            
            for i, cand in enumerate(candidates):
                img_bytes = generate_spectrum_image_bytes(np.array(cand["spectrum"]), title=f"{cand['label']} Sample {cand['sclk']}")
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                prompt = ANNOTATION_USER_PROMPT.format(label=cand['label'])
                
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
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION_TEXT
                        )
                    )
                    explanation = response.text.strip()
                except Exception as e:
                    explanation = f"Failed to generate explanation: {e}"
                    
                yield log_and_yield(self.name, f"Annotated {i+1}/{len(candidates)}: {cand['label']} (sclk: {cand['sclk']})")
                
                new_example = {
                    "label": cand["label"],
                    "image": img_bytes,
                    "explanation": explanation,
                    "sclk": cand["sclk"]
                }
                ctx.session.state["few_shot_examples"].append(new_example)
                
                # Save the annotated spectrum as a PNG file
                import re
                safe_class = re.sub(r'[^\w\s-]', '', cand['label']).replace(' ', '_')
                png_filename = f"cda_dust_agent/data/annotated_spectra/{cand['sclk']}_{safe_class}.png"
                with open(png_filename, "wb") as f:
                    f.write(img_bytes)
                
                # ALSO store in data/input/store
                store_png_filename = f"cda_dust_agent/data/input/store/{cand['sclk']}_{safe_class}.png"
                with open(store_png_filename, "wb") as f:
                    f.write(img_bytes)
                    
                store_json_filename = f"cda_dust_agent/data/input/store/{cand['sclk']}_{safe_class}.json"
                with open(store_json_filename, "w") as f:
                    json.dump({
                        "sclk": cand["sclk"],
                        "label": cand["label"],
                        "prompt": prompt,
                        "explanation": explanation
                    }, f, indent=2)
                
                save_ex = new_example.copy()
                save_ex["image_base64"] = img_b64
                del save_ex["image"]
                
                with open(cache_file, "a") as f:
                    f.write(json.dumps(save_ex) + "\n")
                    
            ctx.session.state["fsa_state"] = "done"
            ctx.session.state["used_ids"] = [ex["sclk"] for ex in ctx.session.state["few_shot_examples"]]
            yield log_and_yield(self.name, f"Automatic few-shot annotations complete. Sent {len(candidates)} spectra to Gemini for few-shot learning. Saved to cache.")
            return
