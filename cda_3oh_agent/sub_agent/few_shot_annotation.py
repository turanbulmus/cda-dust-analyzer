import os
import json
import base64
import asyncio
import logging
from typing import AsyncGenerator
from typing_extensions import override

import pandas as pd
import numpy as np
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import Client
from google.genai.types import Content, Part

from ..config import Config
from ..prompts import (
    FEW_SHOT_ANNOTATION_PROMPT,
    CLASS_PROFILES_3OH,
    CATALOGUE_SUBCLASS_DESCRIPTIONS,
    get_active_system_prompt,
)
from ..tools.utils import generate_spectrum_image_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def log_and_yield(author: str, text: str):
    logger.info(f"[{author}] {text}")
    return Event(author=author, content=Content(parts=[Part.from_text(text=text)]))


configs = Config()


class FewShotAnnotationAgent(BaseAgent):
    """Generates contrastive time-of-flight explanations for the 18 stratified 3-OH / non-3-OH exemplars
    using Gemini 4 Argon and the Referee-optimized system prompt."""

    project_id: str
    location: str
    model_id: str

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        inference_path = configs.agent_settings.inference_path
        if inference_path not in ["local", "batch"]:
            yield log_and_yield(
                self.name, f"Invalid inference_path: {inference_path}. Must be 'local' or 'batch'."
            )
            ctx.session.state["fsa_state"] = "failed"
            return

        cache_path = "cda_3oh_agent/data/input/examples/cached_examples.jsonl"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        force_new = configs.agent_settings.force_new_annotations
        n_Total = configs.agent_settings.few_shot_n

        if n_Total <= 0:
            yield log_and_yield(
                self.name, f"few_shot_n is {n_Total}. Skipping few-shot annotation (Zero-shot mode)."
            )
            ctx.session.state["examples"] = []
            ctx.session.state["used_ids"] = []
            ctx.session.state["fsa_state"] = "done"
            return

        examples = []
        used_ids = []

        if os.path.exists(cache_path) and not force_new:
            yield log_and_yield(self.name, f"Loading cached few-shot examples from {cache_path}...")
            with open(cache_path, "r") as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
            used_ids = [ex["id"] for ex in examples]
            if len(examples) >= n_Total:
                examples = examples[:n_Total]
                used_ids = [ex["id"] for ex in examples]
                yield log_and_yield(self.name, f"Loaded {len(examples)} cached examples.")
                ctx.session.state["examples"] = examples
                ctx.session.state["used_ids"] = used_ids
                ctx.session.state["fsa_state"] = "done"
                return

        yield log_and_yield(
            self.name,
            f"Generating fresh 3-OH / non-3-OH exemplar explanations via {self.model_id}...",
        )
        try:
            df = pd.read_parquet("cda_3oh_agent/data/raw/cda_train.parquet")
            if len(df) > n_Total:
                df = df.iloc[:n_Total].copy()

            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
            os.environ["GOOGLE_API_USE_MTLS"] = "never"
            os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
            client = Client(vertexai=True, project=self.project_id, location="global")

            system_instruction = get_active_system_prompt()
            sem = asyncio.Semaphore(6)

            async def annotate_exemplar(idx: int, row):
                async with sem:
                    sclk_id = str(int(row["sclk"]))
                    binary_label = str(row["class"])
                    cat_class = str(row.get("catalogue_class", binary_label))
                    cat_desc = CATALOGUE_SUBCLASS_DESCRIPTIONS.get(cat_class, cat_class)
                    profile = CLASS_PROFILES_3OH.get(binary_label, "")
                    spectrum = np.array(row["spectrum"], dtype=float)

                    img_bytes = generate_spectrum_image_bytes(spectrum, title=f"Sample {sclk_id}")
                    b64_img = base64.b64encode(img_bytes).decode("utf-8")

                    prompt = FEW_SHOT_ANNOTATION_PROMPT.format(
                        system_instruction=system_instruction,
                        binary_label=binary_label,
                        catalogue_class=cat_class,
                        catalogue_desc=cat_desc,
                    )

                    response = await client.aio.models.generate_content(
                        model=self.model_id,
                        contents=[
                            prompt,
                            Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        ],
                    )
                    raw_expl = response.text.strip().replace("\n", " ")
                    formatted_text = (
                        f"Reference {idx + 1}: {binary_label} (catalogue class '{cat_class}': {cat_desc})\n"
                        f"- Class profile: {profile}\n"
                        f"- Evidence in this spectrum: {raw_expl}"
                    )
                    logger.info(f"[{self.name}] Annotated Reference {idx + 1} (SCLK {sclk_id}, {binary_label} / {cat_class})")
                    return {
                        "id": sclk_id,
                        "sclk": int(row["sclk"]),
                        "label": binary_label,
                        "catalogue_class": cat_class,
                        "explanation": raw_expl,
                        "formatted_text": formatted_text,
                        "image_b64": b64_img,
                    }

            tasks = [annotate_exemplar(i, row) for i, (_, row) in enumerate(df.iterrows())]
            examples = list(await asyncio.gather(*tasks))
            used_ids = [ex["id"] for ex in examples]

            with open(cache_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            yield log_and_yield(self.name, f"Saved {len(examples)} Gemini 4 Argon exemplar annotations to {cache_path}.")
            ctx.session.state["examples"] = examples
            ctx.session.state["used_ids"] = used_ids
            ctx.session.state["fsa_state"] = "done"

        except Exception as e:
            yield log_and_yield(self.name, f"Error during few-shot annotation: {e}")
            ctx.session.state["fsa_state"] = "failed"
            raise e
