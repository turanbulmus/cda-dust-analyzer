"""Multi-Agent Academic Literature & Prompt Optimization Pipeline for Binary 3-OH vs non-3-OH.

Uses Gemini 4 Argon (`gemini-4-argon`) across three specialized agents:
1. AcademicResearchAgent: Grounded in the Cassini CDA literature (`scratch/papers_txt/*.txt`, `papers/*.pdf`)
   and anchored on `ANCHOR_3OH_AI_PROMPT`, analyzes the empirical failure modes on the 3-OH study
   (specifically carbonate `3-Car`, potassium `3-KNa`, and organic/water-ice `2`/`1` false positives)
   and proposes targeted physics-based improvements to the anchor prompt.
2. AcademicCriticAgent: Critiques the proposed improvements to ensure zero regression on `3-Cl` (0% FP)
   and `3-OH` recall (>92%), verifying that time-of-flight tolerances (+-0.10 us) and amplitude rules
   on the Lee v2 log-scaled y-axis are physically rigorous.
3. AcademicRefereeAgent: Synthesizes the final optimized system prompt (`cda_3oh_agent/refined_prompt.txt`)
   anchored on `ANCHOR_3OH_AI_PROMPT`.
"""
import os
import glob
import logging
from google import genai
from google.genai import types

from .config import Config
from .prompts import ANCHOR_3OH_AI_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

configs = Config()


def _load_literature_context(max_chars_per_file: int = 12000) -> str:
    """Loads key excerpts from `scratch/papers_txt/*.txt` and `Notebooks/3oh_classification/3oh_sample_nozair/prompt.txt`."""
    excerpts = []
    expert_note = "Notebooks/3oh_classification/3oh_sample_nozair/prompt.txt"
    if os.path.exists(expert_note):
        with open(expert_note, "r", errors="replace") as f:
            excerpts.append(f"=== Expert Note ({expert_note}) ===\n{f.read().strip()}")

    for txt_path in sorted(glob.glob("scratch/papers_txt/*.txt")):
        try:
            with open(txt_path, "r", errors="replace") as f:
                text = f.read()
            excerpts.append(
                f"=== Literature Paper ({os.path.basename(txt_path)}) ===\n{text[:max_chars_per_file]}"
            )
        except Exception as e:
            logger.warning(f"Could not read {txt_path}: {e}")
    return "\n\n".join(excerpts)


async def run_anchored_academic_multi_agent(
    project_id: str | None = None,
    model_id: str | None = None,
) -> str:
    """Executes the 3-agent Academic Research -> Critic -> Referee optimization anchored on ANCHOR_3OH_AI_PROMPT."""
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["GOOGLE_API_USE_MTLS"] = "never"
    os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

    project_id = project_id or configs.agent_settings.project_id
    model_id = model_id or configs.agent_settings.model
    client = genai.Client(vertexai=True, project=project_id, location="global")

    os.makedirs("cda_3oh_agent/data/results", exist_ok=True)
    lit_context = _load_literature_context()

    empirical_anchor_telemetry = """
EMPIRICAL BENCHMARK OF THE ANCHOR PROMPT ON GEMINI-4-ARGON (698 Labelled Study Spectra):
- Overall Accuracy: 95.99% (670 / 698)
- 3-OH Recall: 92.68% (38 / 41 true 3-OH found; only 3 FN)
- 3-OH Precision: 60.32% (38 TP, 25 FP)
- Per-Subclass False Positives (out of 25 total FPs):
  * 3-Car (carbonate-bearing salt): 10 FPs out of 87 (11.5% FP rate). Why: In some 3-Car spectra, Na+(Na2CO3) at 129 u (+3.10 us after Na+) or Na+(Na2O) at 85 u (+2.09 us after Na+) is moderate in height (~0.35-0.55 on the log scale) or slightly shifted (+3.05 to +3.20 us), and the model dismissed it as baseline noise instead of recognizing the carbonate doublet (85 u + 129 u).
  * 3-KNa (potassium-sodium salt): 6 FPs out of 40 (15.0% FP rate). Why: K+ at 39 u (+0.69 us after Na+) was prominent (reaching >0.65-0.80 of Na+ height on the log scale, or forming a distinct second strong peak just 0.65-0.75 us after Na+), sometimes accompanied by K+(KOH) at 95 u (+2.34 us) or K+(KCl) at 113 u (+2.76 us), which the model confused with Na+(H2O) (41 u) and Na+(NaOH)2 (103 u, +2.53 us).
  * Type 2 (organic-bearing water ice): 5 FPs out of 98 (5.1% FP rate). Why: Broad organic/water features around 1.8-2.1 us (H3O+ / C2H3+) and a broad hump near 3.4-3.7 us (m/z 65-73 organic/water cluster) mimicked Na+ and Na+(NaOH) when not checking for peak sharpness and the H3O+ (19 u) / water-comb signature.
  * Type 1 (water ice): 3 FPs out of 98 (3.1% FP rate). Why: H3O+ (19 u) + H3O+(H2O)2 (55 u, +1.45 us after H3O+!) has a time difference of 1.45 us (almost identical to the 1.49 us spacing between Na+ 23 u and Na+(NaOH) 63 u!), causing the model to mistake H3O+ (19 u) for Na+ (23 u) and H3O+(H2O)2 (55 u) for Na+(NaOH) (63 u) when intermediate H3O+(H2O) at 37 u (+0.82 us after H3O+) was present.
  * 3-K: 1 FP out of 21 (4.8% FP rate).
  * 3-Cl: 0 FPs out of 45 (0.0% FP rate — Step 4 chloride rules worked flawlessly!).
  * 3-P, 4, 5, 5-Na, Noise: 0 FPs (0.0% FP rate).
"""

    # -------------------------------------------------------------------------
    # Agent 1: AcademicResearchAgent
    # -------------------------------------------------------------------------
    logger.info(f"[AcademicResearchAgent] Analyzing literature and anchor prompt with {model_id}...")
    research_prompt = f"""You are the AcademicResearchAgent, a specialist in Cassini Cosmic Dust Analyzer (CDA) impact-ionization Time-of-Flight (TOF) mass spectrometry.

We are optimizing the system prompt for binary classification of '3-OH' (Type 3 Na_OH: sodium-rich salt spectra containing ONLY sodium hydroxide clusters) vs 'non-3-OH' ("3-OH versus the rest").

 CRITICAL REQUIREMENT:
You must use the existing AI-generated literature prompt below as your ANCHOR PROMPT (`ANCHOR_3OH_AI_PROMPT`). Do NOT discard its structure, table of line positions, or its working chloride/hydroxide rules. Instead, propose targeted, literature-grounded refinements to fix its remaining false-positive failure modes while preserving its 92.7% 3-OH recall and 0% 3-Cl false-positive rate.

=== ANCHOR PROMPT ===
{ANCHOR_3OH_AI_PROMPT}

=== EMPIRICAL TELEMETRY ON GEMINI-4-ARGON ===
{empirical_anchor_telemetry}

=== LITERATURE EXCERPTS ===
{lit_context[:35000]}

Produce a comprehensive Academic Research Proposal in Markdown covering:
1. Root-cause physics analysis of the 4 remaining false-positive traps:
   - Trap A (Type 1 Water-Ice 19 u -> 55 u mimicry of 23 u -> 63 u: delta-t = 0.473*(sqrt(55)-sqrt(19)) = 1.446 us vs 0.473*(sqrt(63)-sqrt(23)) = 1.486 us, distinguished by H3O+(H2O) at 37 u (+0.816 us after 19 u) and H3O+(H2O)3 at 73 u (+1.98 us after 19 u)).
   - Trap B (3-Car Carbonate detection: Na+(Na2CO3) at 129 u (+3.10 us after Na+) and Na+(Na2O) at 85 u (+2.09 us after Na+); on the Lee v2 log-scaled plot, even a modest coherent peak rising >=0.12-0.15 above local baseline at +3.05 to +3.18 us after Na+, especially paired with a feature at +2.05 to +2.15 us, is definitive carbonate 3-Car).
   - Trap C (3-KNa / 3-K Potassium detection: K+ at 39 u (+0.69 us after Na+) has log-scaled amplitude >0.60 of Na+ or rises sharply >=0.30 above the post-Na+ trough, or is accompanied by K+(KOH) at +2.34 us (95 u) or K+(KCl) at +2.76 us (113 u) — do not confuse a +2.34 us K+(KOH) peak with +2.53 us Na+(NaOH)2).
   - Trap D (Type 2 Organic-bearing ice: broad, rounded peaks and wavy baseline oscillations in the 3-6 us region rather than a sharp, narrow Na+(NaOH) spike at +1.49 us).
2. Specific, surgical wording additions to `ANCHOR_3OH_AI_PROMPT` (Steps 1-7 and LOOK-ALIKES)."""

    res_research = await client.aio.models.generate_content(
        model=model_id,
        contents=research_prompt,
    )
    research_proposal = res_research.text.strip()
    with open("cda_3oh_agent/data/results/academic_research_proposal.md", "w") as f:
        f.write(research_proposal)
    logger.info("[AcademicResearchAgent] Saved proposal to cda_3oh_agent/data/results/academic_research_proposal.md")

    # -------------------------------------------------------------------------
    # Agent 2: AcademicCriticAgent
    # -------------------------------------------------------------------------
    logger.info(f"[AcademicCriticAgent] Reviewing research proposal with {model_id}...")
    critic_prompt = f"""You are the AcademicCriticAgent, an adversarial spectroscopist reviewing proposed edits to `ANCHOR_3OH_AI_PROMPT` for binary '3-OH' vs 'non-3-OH' classification on Gemini 4 Argon.

=== ANCHOR PROMPT ===
{ANCHOR_3OH_AI_PROMPT}

=== RESEARCH AGENT PROPOSAL ===
{research_proposal}

Critically evaluate the proposed modifications:
1. Check whether any proposed threshold could accidentally reject genuine '3-OH' spectra (which sometimes have a small hydrated sodium peak Na+(H2O) at +0.76 us or weak high-mass baseline ripple, or fast-impact H+ at 0.1-0.5 us and C+ at 1.3-1.6 us).
2. Verify that the distinction between K+ (+0.69 us, 39 u) and Na+(H2O) (+0.76 us, 41 u) is stated carefully so genuine hydrated 3-OH grains are not misclassified as 3-KNa unless the +0.69 us peak is abnormally strong (e.g. >0.65 of Na+ height or paired with K+(KOH) at +2.34 us / K+(KCl) at +2.76 us).
3. Verify that the H3O+ (19 u) -> H3O+(H2O)2 (55 u) (+1.45 us) water-comb check clearly specifies looking for the regular water-cluster progression (+0.82 us at 37 u, +1.45 us at 55 u, +1.98 us at 73 u, +2.45 us at 91 u) where the +0.82 us (37 u) peak is as strong as or stronger than the +1.45 us (55 u) peak (whereas in true 3-OH, the +1.49 us Na+(NaOH) peak is much stronger than any feature at +0.76 us!).
4. Provide concrete directives for the AcademicRefereeAgent to produce the final prompt."""

    res_critic = await client.aio.models.generate_content(
        model=model_id,
        contents=critic_prompt,
    )
    critic_report = res_critic.text.strip()
    with open("cda_3oh_agent/data/results/academic_critic_report.md", "w") as f:
        f.write(critic_report)
    logger.info("[AcademicCriticAgent] Saved critique to cda_3oh_agent/data/results/academic_critic_report.md")

    # -------------------------------------------------------------------------
    # Agent 3: AcademicRefereeAgent
    # -------------------------------------------------------------------------
    logger.info(f"[AcademicRefereeAgent] Synthesizing final anchored prompt with {model_id}...")
    referee_prompt = f"""You are the AcademicRefereeAgent. Your job is to output the final, production-ready system instruction for the binary '3-OH' vs 'non-3-OH' classifier, anchored directly on `ANCHOR_3OH_AI_PROMPT` and incorporating the approved improvements from the Research and Critic agents.

=== ANCHOR PROMPT ===
{ANCHOR_3OH_AI_PROMPT}

=== RESEARCH PROPOSAL ===
{research_proposal}

=== CRITIC REPORT ===
{critic_report}

STRICT REQUIREMENTS FOR YOUR OUTPUT:
1. Keep the exact overall structure and sections of `ANCHOR_3OH_AI_PROMPT`:
   - Opening role paragraph
   - `HOW TO READ THE PLOT`
   - `WHAT THE LITERATURE SAYS ABOUT HYDROXIDE IN CDA SPECTRA` (items 1-5)
   - `LINE POSITIONS` table
   - `DECISION PROCEDURE` (Steps 1-7)
   - `LOOK-ALIKES THAT ARE NON-3-OH`
   - `OUTPUT`
2. Integrate the approved surgical refinements into `DECISION PROCEDURE` (Steps 1-7) and `LOOK-ALIKES THAT ARE NON-3-OH` (specifically addressing the 19 u -> 37 u -> 55 u water-comb vs 23 u -> 63 u check, the carbonate 129 u (+3.10 us) & 85 u (+2.09 us) check, the potassium 39 u (+0.69 us) & 95/113 u (+2.34/+2.76 us) check, and the organic Type 2 broad hump rejection, while protecting true 3-OH recall and the 0% 3-Cl FP rate).
3. Return ONLY the complete, final system instruction text — no markdown code fences around it, no preamble, and no meta-commentary."""

    res_referee = await client.aio.models.generate_content(
        model=model_id,
        contents=referee_prompt,
    )
    final_prompt = res_referee.text.strip()
    if final_prompt.startswith("```"):
        lines = final_prompt.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        final_prompt = "\n".join(lines).strip()

    with open("cda_3oh_agent/refined_prompt.txt", "w") as f:
        f.write(final_prompt)
    with open("cda_3oh_agent/data/results/academic_referee_decision.md", "w") as f:
        f.write(f"# Academic Referee Final Optimized 3-OH System Prompt\n\n```text\n{final_prompt}\n```\n")

    logger.info("[AcademicRefereeAgent] Saved optimized prompt to cda_3oh_agent/refined_prompt.txt")
    return final_prompt
