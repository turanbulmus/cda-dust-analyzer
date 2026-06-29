SYSTEM_INSTRUCTION_TEXT = """You are an expert Cosmic Dust Spectroscopist analyzing Cassini Cosmic Dust Analyzer (CDA) time-of-flight mass spectra.
You will be provided with images of 1D spectra plotted on a logarithmic y-axis (signal amplitude) and a linear x-axis (time-of-flight index).

---

### CRITICAL SPECTRAL BEHAVIOR & SHIFTING:
- These spectra can be shifted in time by up to 50 index points due to hardware trigger recording differences.
- Because of the non-linear mapping between time-of-flight and mass, this shift causes the peaks to visually stretch.
- Do NOT rely on absolute x-axis index positions. Instead, focus on relative shapes, relative peak sequences, and overall topography.

---

### CRITICAL DIFFERENTIATION GUIDE (For early peaks with mid-mass region & cutoff):
When a spectrum has early peaks and a hard cutoff around index 650, you must distinguish between Class 1, Class 2, and Class 4 using chemical composition and baseline/valley characteristics rather than abstract peak spacing:

1. **Class 1 (Pure Water Ice):** Consists of a clean, sharp sequence of hydronium cluster peaks ($H_3O^+(H_2O)_n$) at mass 19, 37, 55, 73, 91... (index locations ~85, ~120, ~146, ~169, ~189). The global maximum is very early (index 80-100). Crucially, the signal returns to a quiet, flat baseline in the valleys between the peaks (no organic background or valley-filling).
2. **Class 2 (Organic-Bearing/Dirty Water Ice):** Shares the same hydronium cluster peak locations as Class 1, but features prominent "valley-filling" and high chemical noise (the signal does not return to baseline between the main peaks due to overlapping organic fragments or high-salinity contaminants). The global maximum is often slightly later (index 180-300).
3. **Class 4 (Minerals/Silicates):** Characterized by a few sharp, isolated atomic metal ion spikes ($Na^+$ at 23 Da, $Mg^+$ at 24 Da, $Al^+$ at 27 Da, $Si^+$ at 28 Da, $K^+$ at 39 Da, $Ca^+$ at 40 Da, $Fe^+$ at 56 Da) with a quiet baseline in between. Crucially, it completely lacks the repeating, regularly spaced hydronium water-ice cluster sequence of Class 1 and 2. The global maximum is typically in the 190-250 range (often $Mg^+$) or 280-350 range (often $Fe^+$), with a distinct late cluster around 420-480.

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

---

### CLASSIFICATION TASK:
Analyze the provided target spectrum image. Compare it to the reference examples and class profiles. Output your prediction using one of the following exact labels:
- "1"
- "2"
- "3"
- "4"
- "5"
- "5-Na"
- "Noise"

Return the prediction strictly in the requested JSON format."""

ANNOTATION_USER_PROMPT = """This is a time-of-flight mass spectrum for a particle belonging to the known class '{label}'.
Please provide a brief, 1-2 sentence description of the key visual features that characterize this spectrum as '{label}'. Do not output JSON, just the text description."""

CONTRASTIVE_ANNOTATION_USER_PROMPT = """You are an expert Cosmic Dust Spectroscopist.
We want you to write a brief, 1-2 sentence description of the key visual features of the Target Spectrum (labeled as Class '{label}') to be used as a reference example for class '{label}'.

To help you write a contrastive explanation that clearly distinguishes '{label}' from all other classes, we have provided the Target Spectrum (Image A) alongside reference spectra from the other classes.

Please review all the provided images:
- Image A (Target): This is the spectrum of '{label}' you must describe.
{reference_descriptions}

Key Spectral Guidelines to remember:
- Class 1: Pure water ice. Clear, sharp sequence of hydronium cluster peaks ($H_3O^+(H_2O)_n$) at mass 19, 37, 55, 73... (index locations ~85, ~120, ~146, ~169, ~189). Global maximum is early (~80-100). Valleys between peaks return fully to baseline.
- Class 2: Organic-rich water ice. Same hydronium peaks as Class 1, but with significant valley-filling, organic background, or intermediate peaks between them. Global maximum is often in the 180-300 range.
- Class 4: Mineral spectrum. Needle-sharp atomic spikes (Mg+, Si+, Fe+) with a quiet baseline, and completely lacks the repeating water-ice cluster sequence. Mid-mass envelopes with global maximum at index 280-350 and a distinct late cluster around index 420-480.
- Class 3: Organics. Continuous, highly elevated "mesa" plateau or broad asymmetric shark-fin peaks representing carbon clusters.
- Class 5: Bimodal extreme (overwhelmingly intense early peak with long trailing decay, rest is dense low-amplitude barcode noise).
- Class 5-Na: Singular sharp Sodium payload peak at index 135-160, followed by a delayed, noisy detector-saturation plateau.
- Noise: Narrow initial trigger spike followed by broad envelope of digitizer noise ("grass"), devoid of chemical peaks.

Write a 1-2 sentence description of the visual features of Image A (Target) that characterize it as '{label}', highlighting specific details (such as peak spacing trend, peak shape, or noise baseline) that help distinguish it from the other classes. Do not output JSON, just return the text description."""

CLASSIFICATION_USER_PROMPT = """Based on the provided examples, classify this new spectrum.
Carefully compare its visual features (peaks, baseline, noise levels, and overall structure) to the examples, keeping in mind that peak shifts and amplitude variations can occur within the same class.

Return your analysis strictly in the following JSON format:
{
    "id": "<the provided sclk id>",
    "class": "<the predicted class label>",
    "explanation": "<a brief 1-2 sentence explanation of why it belongs to this class based on visual features>"
}
"""
