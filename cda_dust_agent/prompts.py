SYSTEM_INSTRUCTION_TEXT = """Because of hardware trigger recording differences and variations in impact-induced plasma generation/ion extraction dynamics, these spectra can be shifted in time by up to 50 index points. Due to the non-linear mapping between time-of-flight and mass (t ∝ √m), this shift causes peaks to visually stretch non-linearly. Do NOT rely strictly on absolute x-axis index positions. Instead, use Relative Mass Ratio Anchoring (t2/t1 ≈ √(m2/m1)) relative to prominent early anchor peaks (e.g., H3O+ at 19 Da, Na+ at 23 Da, Mg+ at 24 Da, K+ at 39 Da), and focus on relative shapes, envelope sequences, and overall spectral topography.

MANDATORY DISQUALIFICATION AUDIT (Apply in sequence before assigning final class):

1. RULE 0 (NOISE GATE & LOW-S/N CHECK):
   - If QI_AMPL < 5.0e-14 C OR Peak SNR < 4.0, the spectrum is physically near the instrumental noise floor. Do NOT interpret baseline wander or high-frequency digitizer grass as chemical features or organic envelopes. Classify strictly as Noise unless explicit, resolved chemical peaks exceed SNR > 3.0.

2. RULE 1 (DIGITIZER GRASS VS. ORGANIC CONTINUUM):
   - High-frequency single-pixel digitizer quantization grass MUST NOT be classified as Class 2 organic valley filling or Class 3 envelopes.
   - Class 1 & 2 REQUIRE a regular sequence of hydronium cluster peaks (H3O+(H2O)_n at m/z 19, 37, 55, 73 Da) verified by relative mass ratios.
   - If the hydronium inter-peak valleys drop near the baseline (y < 0.15), classify as Class 1 (Pure Water Ice). Class 2 requires smooth, resolved intermediate VOC peaks or broad background humps.

3. RULE 2 (CLASS 5 ALKALI DOUBLET VERIFICATION):
   - A single sharp early peak followed by flat baseline grass MUST NOT be classified as Class 5 unless verified as Na+ (23 Da) or K+ (39 Da) with characteristic K+/Na+ doublet mass ratios (t_K/t_Na ≈ 1.30).
   - A single early peak at m/z 19 (H3O+) or m/z 1 (H+) with flat baseline MUST be classified as low-signal Class 1, NOT Class 5.

4. RULE 3 (SODIUM ORTHOPHOSPHATE REQUIREMENT FOR CLASS 5-Na):
   - A detector saturation plateau (y > 0.8) alone DOES NOT constitute Class 5-Na.
   - Class 5-Na MANDATES an early Na+ payload spike (23 Da) AND diagnostic sodium orthophosphate cluster peaks at t/t_Na = √(125/23) ≈ 2.33, √(165/23) ≈ 2.68, or √(187/23) ≈ 2.85. If saturation follows an organic precursor envelope without orthophosphate mass clusters, classify as Class 3.

5. RULE 4 (CLASS 3-P EXCLUSION RULE):
   - Class 3-P requires an explosive primary multi-peak complex (global max y = 1.0) and a continuous elevated chemical noise cushion (y ≈ 0.25-0.50).
   - EXCLUSIONS: If repeating hydronium cluster peaks (19, 37, 55 Da) are present, classify as Class 1 or 2, NOT 3-P. If the early peak is a single needle-sharp atomic alkali spike, classify as Class 5 or 5-Na, NOT 3-P. If deep valleys (y < 0.15) separate envelopes, classify as Class 3.

6. RULE 5 (CLASS 4 SILICATE ELEMENTAL ATOMIC SPIKE REQUIREMENT):
   - Class 4 MANDATES narrow, sharp elemental atomic cation spikes in the low-mass region (Mg+ at 24 Da / Si+ at 28 Da / Ca+ at 40 Da with FWHM ≤ 3 channels) preceding mid-mass silicate envelopes. Broad mid-mass envelopes without early Mg+/Si+ spikes indicate Class 3 (Organics).

CLASS SPECIFIC PROFILES:
*   **Class Noise:** Instrumental digitizer noise showing a narrow trigger spike and high-frequency 'grass', completely devoid of chemical peaks or distinct spectral features.
*   **Class 1:** Pure water ice (Type I) spectrum consisting of a regular sequence of hydronium cluster peaks (H3O+(H2O)_n) at mass 19, 37, 55, 73, 91... Early global maximum (near m/z 19 or 37) followed by a rapid, step-like decay of peak heights. Valleys between peaks return completely to the zero baseline without organic background or valley filling. Terminates in a sharp cutoff near index 650.
*   **Class 2:** Organic-bearing or dirty water ice (Type II). Features hydronium cluster peak locations similar to Class 1, but with significant organic or saline contamination filling the inter-peak valleys. Includes low-mass volatile organic compound (VOC) subtypes (nitrogenous 2N, oxygenated 2O, aromatic precursors 2A, mixed 2M). Global maximum often appears in the mid-mass range with broad unresolved cluster sequences.
*   **Class 3:** Macromolecular organic cations (HMOC subset of Type II). Characterized by 3 to 4 broad, asymmetric 'shark-fin' peak clusters with expanding periodicity (12-14 Da carbon series spacing, aromatic/N/O envelopes near ~77-80 Da). Crucially, valleys between main clusters drop significantly (y < 0.15). Water cluster peaks are absent or minor, and no single early peak overwhelmingly dominates the spectrum.
*   **Class 4:** Mineral/silicate-rich spectrum. Isolated, needle-sharp atomic spikes in the low-mass region (Mg+ at 24 Da, Si+ at 28 Da, Ca+ at 40 Da; noting potential Fe depletion in E-ring nanodust) with a quiet baseline in between. Lacks repeating hydronium sequences. Transitioning to broad mid-to-high mass silicate envelopes terminating near index 650.
*   **Class 5:** Sputtered alkali salt cores / altered high-salt grains (Type 5). Bimodal extreme with an overwhelmingly intense primary atomic peak (K+ at 39 Da, Li+ at 7 Da, or Mg+ at 24 Da) with a long trailing decay edge, followed by a dense, low-amplitude noisy barcode/grass background resulting from detector recovery or fragmented ions.
*   **Class 5-Na:** Bipartite sodium chemistry (Type III). Erupts with a singular, overwhelmingly intense sharp spike (Sodium payload Na+ at 23 Da), followed by a delayed, prolonged, highly noisy detector-saturation plateau (high salt content: Na+, K+, Cl-, HCO3-, plus sodium orthophosphates at 125, 165, 187 Da) terminating in a hard cliff near index 650.
*   **Class 3-P:** Burst-and-trail agglomerate signature from porous particle impacts. Shows broad unresolved multi-peak mass envelopes, an explosive primary complex (global max), secondary peak, and a continuous elevated chemical noise baseline cushion past index 400 that stays elevated until a rapid collapse between index 650 and 750."""

SYSTEM_INSTRUCTION_TEXT_SELF_CORRECT = """When evaluating your initial classification, apply the following rigorous verification and exclusion checks:
1. **Class 3-P Multi-Condition Check & False Positive Prevention:**
   * Verify that the spectrum contains an explosive multi-peak primary agglomerate complex and a secondary envelope peak.
   * Verify that the baseline past early features stays elevated (y ≈ 0.25 - 0.50) as a continuous chemical noise cushion.
   * EXCLUSION: If the dominant early feature is a single narrow atomic alkali spike (Na+ at 23 Da or K+ at 39 Da) followed by a flat saturation level or ringing, REJECT Class 3-P and reclassify as Class 5-Na or Class 5.
2. **Class 3 vs Class 2 Valley Depth Check:**
   * For Class 3 (HMOCs), verify that the valleys between broad shark-fin carbon envelopes (12-14 Da spacing) drop below y < 0.15, returning near the baseline.
   * If hydronium clusters (19, 37, 55 Da) dominate with filled valleys, classify as Class 2.
3. **Relative Mass Ratio Verification:**
   * Do not rely on fixed pixel index numbers. Verify peak assignments using Relative Mass Ratio Anchoring (t2/t1 ≈ √(m2/m1)) from early identified anchor ions."""

ANNOTATION_USER_PROMPT = """This is a time-of-flight mass spectrum for a particle belonging to the known class '{label}'.
Please provide a brief, 1-2 sentence description of the key visual features that characterize this spectrum as '{label}'. Do not output JSON, just the text description."""

CLASSIFICATION_USER_PROMPT = """Based on the provided examples, classify this new spectrum.
Carefully compare its visual features (peaks, baseline, noise levels, and overall structure) to the examples, keeping in mind that peak shifts and amplitude variations can occur within the same class.

Return your analysis strictly in the following JSON format:
{
    "id": "<the provided sclk id>",
    "class": "<the predicted class label>",
    "explanation": "<a brief 1-2 sentence explanation of why it belongs to this class based on visual features>"
}
"""

GENERAL_PROFILES = {
    "Noise": "Instrumental digitizer noise showing a narrow trigger spike and high-frequency 'grass', completely devoid of chemical peaks.",
    "1": "Pure water ice (Type I) sequence of regularly spaced hydronium cluster peaks (H3O+(H2O)_n) with an early global maximum and clean valleys returning fully to baseline.",
    "2": "Organic-bearing water ice (Type II) featuring hydronium peaks with filled valleys containing low-mass organic VOC background (nitrogenous, oxygenated, aromatic precursors).",
    "3": "Macromolecular organic cations (HMOCs) showing broad, asymmetric 'shark-fin' carbon cluster envelopes (12-14 Da periodicity) with deep valleys (y < 0.15) and no single dominating early peak.",
    "4": "Mineral/silicate spectrum with isolated, needle-sharp atomic spikes (Mg+, Si+, Ca+, Fe-depleted) on a quiet baseline, transitioning to mid-mass silicate envelopes.",
    "5": "Bimodal extreme from sputtered alkali cores (Type 5) featuring an intense early atomic peak (K+, Li+, Mg+) with trailing decay and a dense, low-amplitude noisy barcode background.",
    "5-Na": "Bipartite sodium chemistry (Type III) dominated by an intense sodium payload spike (Na+ at 23 Da), sodium orthophosphates (125/165/187 Da), and a delayed detector-saturation plateau.",
    "3-P": "Burst-and-trail agglomerate impact signature with an explosive multi-peak primary complex, secondary envelope, and a continuous elevated chemical noise baseline cushion."
}
