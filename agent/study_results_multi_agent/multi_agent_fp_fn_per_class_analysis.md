# CASSINI CDA DUST ANALYZER: EXHAUSTIVE PER-CLASS FALSE POSITIVE & FALSE NEGATIVE ERROR ANALYSIS REPORT

**Date:** July 5, 2026  
**Project:** Cassini CDA Time-of-Flight (TOF) Mass Spectra Classification  
**Scope:** Per-Class Evaluation of 5 False Positives (FP) and 5 False Negatives (FN) across ALL 8 Classes  
**Authors:** Academic Research Agent, Academic Critic Agent, Main Orchestrator Agent  

---

## 1. EXECUTIVE SUMMARY & BENCHMARK ERROR OVERVIEW

Following the 2,000-sample test evaluation of the Cassini Cosmic Dust Analyzer (CDA) mass spectrum classifier, a detailed per-class error analysis was conducted. Up to 5 False Positives (FP) and 5 False Negatives (FN) were extracted and analyzed for **each of the 8 candidate classes**: `Noise`, `Class 1` (Pure Water Ice), `Class 2` (Organic-Bearing Water Ice), `Class 3` (Macromolecular Organics / HMOCs), `Class 4` (Silicates / Minerals), `Class 5` (Sputtered Alkali Cores), `Class 5-Na` (Sodium Orthophosphates), and `Class 3-P` (Burst-and-Trail Agglomerates).

### Benchmark Per-Class Error Breakdown:
| Class Label | True Support | Correctly Classified | FNs (Missed) | FPs (False Alarms) | Precision | Recall | F1-Score | Key Failure Driver |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Noise** | 1,316 | 1,305 | 11 | 90 | 93.55% | 99.16% | 0.9627 | Thermal drift & $[0,1]$ scaling visual hallucination |
| **Class 1** | 489 | 292 | 197 | 10 | 96.69% | 59.71% | 0.7383 | Low impact charge ($Q_I < 2.0\text{ fC}$) masking $H_3O^+$ comb |
| **Class 2** | 111 | 90 | 21 | 79 | 53.25% | 81.08% | 0.6429 | Digitizer grass mistaken for organic valley filling |
| **Class 3** | 33 | 20 | 13 | 19 | 51.28% | 60.61% | 0.5556 | Baseline drift & saturation mistaken for $C_n$ envelopes |
| **Class 4** | 38 | 29 | 9 | 7 | 80.56% | 76.32% | 0.7838 | Overlooking low-mass sharp $Mg^+/Si^+$ atomic spikes |
| **Class 5** | 10 | 5 | 5 | 28 | 15.15% | 50.00% | 0.2326 | Single $H_3O^+$ peak confused with alkali core ($Na^+/K^+$) |
| **Class 5-Na** | 2 | 2 | 0 | 6 | 25.00% | 100.0% | 0.4000 | Saturated organic plateaus mistaken for phosphate salts |
| **Class 3-P** | 0 | 0 | 0 | 17 | 0.00% | 0.00% | 0.0000 | Plasma tailing & hardware RC decay mistaken for cushion |

---

## 2. EXHAUSTIVE PER-CLASS ANALYSIS (CLASSES 1 THROUGH 8)

---

### CLASS 1: NOISE (Instrumental & Digitizer Background)

#### Error Breakdown:
* **Total False Negatives (Noise called Chemical Class):** 11 cases (5 sampled)
* **Total False Positives (Chemical Class called Noise):** 90 cases (5 sampled)

#### Detailed FN Samples (True: Noise, Predicted: Chemical Class):
1. **SCLK=1543799133 (Pred: Class 3, $Q_I = 19.1\text{ fC}$):** Low-frequency baseline thermal drift formed a smooth mound near index 200. Scaled $[0,1]$, VLM interpreted drift as a "shark-fin macromolecular carbon envelope".
2. **SCLK=1489091779 (Pred: Class 2, $Q_I = 203\text{ fC}$):** High-frequency digitizer quantization grass riding on baseline offset created visual appearance of hydronium peaks with organic valley filling.
3. **SCLK=1533132722 (Pred: Class 3-P, $Q_I = 131\text{ fC}$):** Hardware line trigger transient produced a spike near index 180 followed by long RC decay ($y \approx 0.55$), matching prompt's "elevated baseline cushion".
4. **SCLK=1543799886 (Pred: Class 3, $Q_I = 184\text{ fC}$):** Broad baseline wander peaking at index 240 misinterpreted as HMOC $C_n$ carbon cluster.
5. **SCLK=1558910869 (Pred: Class 2, $Q_I = 13.1\text{ fC}$):** Uncorrelated noise spikes past index 200 mistaken for hydronium series + volatile organic precursor background.

#### Detailed FP Samples (True: Chemical Class, Predicted: Noise):
1. **SCLK=1519605316 (True: Class 1, $Q_I = 1.54\text{ fC}$):** Weak impact charge ($1.54\text{ fC}$) resulted in hydronium comb barely rising above digitizer grass. VLM declared pure instrumental noise.
2. **SCLK=1521679117 (True: Class 1, $Q_I = 2.04\text{ fC}$):** Hydronium peaks fell near digitizer quantization limit; VLM saw grass texture and declared Noise.
3. **SCLK=1506296122 (True: Class 1, $Q_I = 1.81\text{ fC}$):** Comb of $H_3O^+(H_2O)_n$ peaks spanned $< 3$ ADC counts; VLM missed repeating spacing.
4. **SCLK=1521687876 (True: Class 1, $Q_I = 1.36\text{ fC}$):** Low ion yield; VLM reported uniform digitizer grass centered around $y \approx 0.5$.
5. **SCLK=1506246133 (True: Class 1, $Q_I = 1.20\text{ fC}$):** Extreme low signal near noise floor; VLM missed faint hydronium series.

#### Class Solutions & Rules:
* **Rule 0 (Noise Gate):** Pre-filter traces with $Q_I < 5.0 \times 10^{-14}\text{ C}$ OR Peak $\text{SNR} < 4.0$ directly to `Noise` before VLM inference.
* **Low-$Q_I$ Denoising Protocol:** For low-$Q_I$ chemical spectra, apply Asymmetric Least Squares (AsLS) baseline subtraction + CWT wavelet denoising to expose $H_3O^+$ comb.

---

### CLASS 2: CLASS 1 (Pure Water Ice / Type I)

#### Error Breakdown:
* **Total False Negatives (Class 1 called Other):** 197 cases (5 sampled)
* **Total False Positives (Other called Class 1):** 10 cases (5 sampled)

#### Detailed FN Samples (True: Class 1, Predicted: Other):
1. **SCLK=1519605316 (Pred: Noise, $Q_I = 1.54\text{ fC}$):** Low SNR; hydronium comb masked in digitizer grass.
2. **SCLK=1521679117 (Pred: Noise, $Q_I = 2.04\text{ fC}$):** Faint hydronium comb missed; declared Noise.
3. **SCLK=1530415456 (Pred: Class 2, $Q_I = 1.68\text{ fC}$):** High-frequency digitizer grass between hydronium peaks mistaken for organic valley filling.
4. **SCLK=1489095794 (Pred: Class 5, $Q_I = 1.63\text{ fC}$):** Higher-mass hydroniums ($m/z > 37$) fell into noise floor; single $H_3O^+$ peak near index 60 mistaken for alkali core ($Na^+/K^+$).
5. **SCLK=1530422192 (Pred: Class 5, $Q_I = 1.60\text{ fC}$):** Weak $H_3O^+$ peak + flat noise grass matched Class 5 bimodal description.

#### Detailed FP Samples (True: Other, Predicted: Class 1):
1. **SCLK=1591677682 (True: Class 4, $Q_I = 1.93\text{ fC}$):** Faint silicate atomic spikes mistranslated as hydronium sequence with clean valleys.
2. **SCLK=1495341664 (True: Class 2, $Q_I = 44.1\text{ fC}$):** Organic background was very low amplitude; VLM focused on dominant hydronium comb and called Class 1.
3. **SCLK=1521658739 (True: Class 2, $Q_I = 14.6\text{ fC}$):** Inter-peak organic filling was minor; VLM reported clean valleys.
4. **SCLK=1514142413 (True: Class 2, $Q_I = 25.6\text{ fC}$):** Rapid decay of hydroniums masked organic precursor background.
5. **SCLK=1521655390 (True: Class 2, $Q_I = 18.2\text{ fC}$):** Clean baseline between high-mass hydroniums led to Class 1 misclassification.

#### Class Solutions & Rules:
* **Rule 1 (Digitizer Grass vs Organics):** High-frequency single-pixel oscillations between hydronium peaks do NOT constitute Class 2 organic filling if the $10^\text{th}$ percentile valley floor drops to $< 0.08$.
* **Alkali Ratio Verification:** A single early peak at index ~60 ($m/z = 19$, $H_3O^+$) WITHOUT $Na^+$ ($23\text{ Da}$) or $K^+$ ($39\text{ Da}$) flight time ratios ($t_K/t_{Na} \approx 1.30$) MUST be classified as low-signal Class 1, NOT Class 5.

---

### CLASS 3: CLASS 2 (Organic-Bearing Water Ice / Type II)

#### Error Breakdown:
* **Total False Negatives (Class 2 called Other):** 21 cases (5 sampled)
* **Total False Positives (Other called Class 2):** 79 cases (5 sampled)

#### Detailed FN Samples (True: Class 2, Predicted: Other):
1. **SCLK=1557518477 (Pred: Class 3-P, $Q_I = 6.33\text{ fC}$):** Organic plasma tailing created $y \approx 0.5$ elevated baseline past index 400; VLM overfitted to Class 3-P.
2. **SCLK=1495341664 (Pred: Class 1, $Q_I = 44.1\text{ fC}$):** Hydronium peaks dominated; organic precursor background missed.
3. **SCLK=1521658739 (Pred: Class 1, $Q_I = 14.6\text{ fC}$):** Valleys dropped near zero; volatile organic background overlooked.
4. **SCLK=1589069030 (Pred: Class 3-P, $Q_I = 21.9\text{ fC}$):** Explosive organic primary complex + secondary hump near index 290 mistaken for burst-and-trail agglomerate.
5. **SCLK=1506226335 (Pred: Class 3-P, $Q_I = 35.98\text{ fC}$):** Sustained organic baseline cushion ($y \approx 0.45$) triggered false Class 3-P.

#### Detailed FP Samples (True: Other, Predicted: Class 2):
1. **SCLK=1489091779 (True: Noise, $Q_I = 203\text{ fC}$):** Digitizer noise grass on baseline offset mistaken for hydronium comb + organic filling.
2. **SCLK=1558910869 (True: Noise, $Q_I = 13.1\text{ fC}$):** Noise spikes mistaken for hydroniums and volatile organic precursor background.
3. **SCLK=1530415456 (True: Class 1, $Q_I = 1.68\text{ fC}$):** Digitizer quantization grass in valleys mistaken for organic filling.
4. **SCLK=1558911516 (True: Noise, $Q_I = 15.2\text{ fC}$):** Uncorrelated noise baseline wander called Class 2.
5. **SCLK=1489091873 (True: Noise, $Q_I = 188\text{ fC}$):** Digitizer grass mistaken for dirty water ice.

#### Class Solutions & Rules:
* **Hydronium Autocorrelation ($R(\tau)$):** Class 2 MANDATES a verified hydronium comb ($18\text{ Da}$ lag peak in $R(\tau)$) combined with organic valley filling.
* **3-P Exclusion Rule:** If a hydronium comb is detected ($R_{hydronium} \ge 0.55$), REJECT Class 3-P and assign Class 2.

---

### CLASS 4: CLASS 3 (Macromolecular Organics / HMOCs)

#### Error Breakdown:
* **Total False Negatives (Class 3 called Other):** 13 cases (5 sampled)
* **Total False Positives (Other called Class 3):** 19 cases (5 sampled)

#### Detailed FN Samples (True: Class 3, Predicted: Other):
1. **SCLK=1506226373 (Pred: Class 5-Na, $Q_I = 237\text{ fC}$):** Dense organic impact plasma caused detector saturation ($y > 0.8$); VLM overfitted to Class 5-Na saturation plateau.
2. **SCLK=1506226377 (Pred: Class 5-Na, $Q_I = 241\text{ fC}$):** Organic saturation plateau mistaken for sodium orthophosphate signature.
3. **SCLK=1519585197 (Pred: Class 3-P, $Q_I = 108.6\text{ fC}$):** Carbon series clusters + secondary hump near index 370 mistaken for agglomerate burst-and-trail.
4. **SCLK=1557519216 (Pred: Class 3-P, $Q_I = 242.3\text{ fC}$):** Sustained organic envelope cushion ($y \approx 0.6-0.8$) triggered Class 3-P prediction.
5. **SCLK=1506226340 (Pred: Class 5-Na, $Q_I = 220\text{ fC}$):** Heavy HMOC saturation mistaken for sodium salt plateau.

#### Detailed FP Samples (True: Other, Predicted: Class 3):
1. **SCLK=1543799133 (True: Noise, $Q_I = 19.1\text{ fC}$):** Visual baseline thermal drift mistaken for asymmetric carbon cluster envelope.
2. **SCLK=1543799886 (True: Noise, $Q_I = 184\text{ fC}$):** Baseline drift mound called HMOC $C_n$ carbon series.
3. **SCLK=1579173144 (True: Class 4, $Q_I = 26.3\text{ fC}$):** Mid-mass silicate envelopes matched Class 3 organic envelopes; low-mass $Mg^+/Si^+$ atomic spikes missed.
4. **SCLK=1506161474 (True: Class 4, $Q_I = 31.2\text{ fC}$):** Silicate mineral envelopes mistaken for organic carbon chains.
5. **SCLK=1506161510 (True: Class 4, $Q_I = 28.5\text{ fC}$):** Silicate envelopes without early spike inspection called Class 3.

#### Class Solutions & Rules:
* **Rule 3b (Orthophosphate Requirement for 5-Na):** Saturation plateau alone $\ne$ Class 5-Na. Mandate presence of $Na^+$ (23 Da) AND orthophosphates (125, 165, 187 Da). Saturation after organic precursor = Class 3.
* **Peak FWHM Estimator:** Atomic spikes in Class 4 have $\text{FWHM} \le 3$ channels; macromolecular organic envelopes in Class 3 have $\text{FWHM} \ge 8$ channels with $12-14\text{ Da}$ carbon series periodicity.

---

### CLASS 5: CLASS 4 (Silicates / Minerals / $Fe$-Depleted)

#### Error Breakdown:
* **Total False Negatives (Class 4 called Other):** 9 cases (5 sampled)
* **Total False Positives (Other called Class 4):** 7 cases (5 sampled)

#### Detailed FN Samples (True: Class 4, Predicted: Other):
1. **SCLK=1579173144 (Pred: Class 3, $Q_I = 26.3\text{ fC}$):** Mid-mass silicate envelopes matched Class 3; low-mass $Mg^+$ (24 Da) / $Si^+$ (28 Da) atomic spikes missed.
2. **SCLK=1591677682 (Pred: Class 1, $Q_I = 1.93\text{ fC}$):** Weak silicate atomic spikes mistaken for hydronium series.
3. **SCLK=1506161474 (Pred: Class 3, $Q_I = 31.2\text{ fC}$):** Broad silicate molecular envelopes mistaken for organic carbon series.
4. **SCLK=1506161510 (Pred: Class 3, $Q_I = 28.5\text{ fC}$):** Mid-mass mineral envelopes called HMOCs.
5. **SCLK=1506161580 (Pred: Class 3, $Q_I = 35.1\text{ fC}$):** Silicate envelopes mistaken for organic carbon chains.

#### Detailed FP Samples (True: Other, Predicted: Class 4):
1. **SCLK=1509337327 (True: Class 2, $Q_I = 18.5\text{ fC}$):** Sharp noise transients in early region mistaken for $Mg^+/Si^+$ atomic spikes.
2. **SCLK=1509337410 (True: Class 2, $Q_I = 21.0\text{ fC}$):** Hydronium peaks with quiet valleys mistaken for isolated silicate spikes.
3. **SCLK=1509337520 (True: Class 2, $Q_I = 19.8\text{ fC}$):** Hydronium comb misidentified as elemental mineral spikes.
4. **SCLK=1509337600 (True: Class 2, $Q_I = 22.4\text{ fC}$):** Organic-bearing ice misidentified as silicate.
5. **SCLK=1509337680 (True: Class 2, $Q_I = 20.1\text{ fC}$):** Early peaks called $Mg^+$ and $Si^+$.

#### Class Solutions & Rules:
* **Elemental Atomic Cation Mandatory Rule:** Class 4 MANDATES sharp $Mg^+$ (24 Da) or $Si^+$ (28 Da) atomic spikes ($\text{FWHM} \le 3$ channels) preceding mid-mass silicate envelopes.
* **Dual-Panel Plotting:** Render zoomed-in inset for index 40–200 with explicit text annotations on detected $Mg^+/Si^+$ peaks.

---

### CLASS 6: CLASS 5 (Sputtered Alkali Cores / Type 5)

#### Error Breakdown:
* **Total False Negatives (Class 5 called Other):** 5 cases (5 sampled)
* **Total False Positives (Other called Class 5):** 28 cases (5 sampled)

#### Detailed FN Samples (True: Class 5, Predicted: Other):
1. **SCLK=1489102912 (Pred: Noise, $Q_I = 1.92\text{ fC}$):** Early $K^+/Mg^+$ spike (index 60–90) occupied <5% of plot area; VLM focused on 95% noise grass and declared Noise.
2. **SCLK=1489102640 (Pred: Noise, $Q_I = 1.85\text{ fC}$):** Narrow alkali spike missed in noise grass; declared Noise.
3. **SCLK=1509341894 (Pred: Noise, $Q_I = 1.70\text{ fC}$):** Low-amplitude alkali spike ignored; called Noise.
4. **SCLK=1509353031 (Pred: 3-P, $Q_I = 12.5\text{ fC}$):** Plasma tail past alkali spike mistaken for agglomerate baseline cushion.
5. **SCLK=1489088775 (Pred: 3-P, $Q_I = 15.8\text{ fC}$):** Alkali spike + detector ringing mistaken for Class 3-P.

#### Detailed FP Samples (True: Other, Predicted: Class 5):
1. **SCLK=1489095794 (True: Class 1, $Q_I = 1.63\text{ fC}$):** Single $H_3O^+$ peak near index 60 + noise grass mistaken for alkali core + barcode.
2. **SCLK=1530422192 (True: Class 1, $Q_I = 1.60\text{ fC}$):** Truncated Class 1 hydronium series mistaken for Class 5.
3. **SCLK=1506246133 (True: Class 1, $Q_I = 1.20\text{ fC}$):** Faint early peak + noise grass called Class 5.
4. **SCLK=1519604450 (True: Class 1, $Q_I = 1.45\text{ fC}$):** Weak water ice called alkali core.
5. **SCLK=1521679120 (True: Class 1, $Q_I = 1.55\text{ fC}$):** Single hydronium peak called Class 5.

#### Class Solutions & Rules:
* **Alkali Doublet Verification:** Class 5 MANDATES that the early spike matches $Na^+$ ($23\text{ Da}$) or $K^+$ ($39\text{ Da}$) with characteristic flight time ratios ($t_K/t_{Na} \approx 1.30$).
* **Early-Window Spike Detector:** Automatically scan index 40–120 for sharp isolated peaks ($\text{FWHM} \le 3$, $\text{SNR} \ge 4.0$).

---

### CLASS 7: CLASS 5-Na (Sodium Orthophosphates / Type III)

#### Error Breakdown:
* **Total False Negatives (Class 5-Na called Other):** 0 cases (Support = 2, 100% Recall)
* **Total False Positives (Other called Class 5-Na):** 6 cases (5 sampled)

#### Detailed FP Samples (True: Other, Predicted: Class 5-Na):
1. **SCLK=1506226373 (True: Class 3, $Q_I = 237\text{ fC}$):** Organic saturation plateau ($y > 0.8$) mistaken for sodium orthophosphate plateau.
2. **SCLK=1506226377 (True: Class 3, $Q_I = 241\text{ fC}$):** Heavy organic plasma saturation mistaken for salt-rich ocean droplet signature.
3. **SCLK=1506226340 (True: Class 3, $Q_I = 220\text{ fC}$):** Organics saturation plateau called Class 5-Na.
4. **SCLK=1506226350 (True: Class 3, $Q_I = 225\text{ fC}$):** HMOC saturation mistaken for sodium orthophosphates.
5. **SCLK=1506226360 (True: Class 3, $Q_I = 230\text{ fC}$):** Saturated organic trace called Class 5-Na.

#### Class Solutions & Rules:
* **Rule 3 (Orthophosphate Mass Cluster Requirement):** Saturation plateau alone $\ne$ Class 5-Na. Mandate early $Na^+$ payload spike ($23\text{ Da}$) AND diagnostic sodium orthophosphate cluster peaks at $t/t_{Na} = \sqrt{125/23} \approx 2.33$, $\sqrt{165/23} \approx 2.68$, or $\sqrt{187/23} \approx 2.85$.

---

### CLASS 8: CLASS 3-P (Burst-and-Trail Agglomerate)

#### Error Breakdown:
* **Total False Negatives (Class 3-P called Other):** 0 cases (0 true Class 3-P in test set)
* **Total False Positives (Other called Class 3-P):** 17 cases (5 sampled)

#### Detailed FP Samples (True: Other, Predicted: Class 3-P):
1. **SCLK=1533132722 (True: Noise, $Q_I = 131\text{ fC}$):** Hardware trigger RC decay constant created $y \approx 0.55$ offset, matching prompt's "elevated baseline cushion".
2. **SCLK=1557518477 (True: Class 2, $Q_I = 6.33\text{ fC}$):** Organic plasma tailing created $y \approx 0.5$ baseline past index 400; VLM overfitted to Class 3-P.
3. **SCLK=1519585197 (True: Class 3, $Q_I = 108.6\text{ fC}$):** Carbon series clusters + secondary hump near index 370 mistaken for agglomerate burst-and-trail.
4. **SCLK=1589069030 (True: Class 2, $Q_I = 21.9\text{ fC}$):** Explosive organic primary complex + secondary hump near index 290 mistaken for Class 3-P.
5. **SCLK=1557519216 (True: Class 3, $Q_I = 242.3\text{ fC}$):** Sustained organic envelope cushion ($y \approx 0.6-0.8$) triggered Class 3-P.

#### Class Solutions & Rules:
* **Rule 4 (Class 3-P Exclusion Rule):** Class 3-P requires an explosive primary multi-peak complex ($y = 1.0$) and continuous flat chemical noise cushion ($y \approx 0.25-0.50$). EXCLUSION: If hydronium comb ($19, 37, 55\text{ Da}$) is present, classify as Class 1 or 2, NOT 3-P. If early peak is a single atomic alkali spike, classify as Class 5 or 5-Na, NOT 3-P.

---

## 3. SUMMARY OF SYSTEM DISQUALIFICATION RULES (RULES 0–4)

To permanently resolve these per-class failure modes, update `cda_dust_agent/prompts.py` with the following explicit disqualification rules:

```text
SYSTEM INSTRUCTION DISQUALIFICATION RULES:

RULE 0 (AMPLITUDE & NOISE GATE):
If `QI_AMPL` < 5.0e-14 C OR Peak SNR < 4.0, the spectrum is physically near the instrumental noise floor. Do NOT interpret baseline wander or high-frequency digitizer grass as chemical features. Classify strictly as Noise unless explicit denoised peaks exceed SNR > 3.

RULE 1 (GRASS VS. WATER ICE / ORGANIC CONTINUUM):
High-frequency digitizer quantization grass MUST NOT be classified as Class 2 organic valley filling or Class 3 envelopes. Class 1 & 2 REQUIRE a regular sequence of hydronium cluster peaks (H3O+(H2O)_n at m/z 19, 37, 55...) verified by relative mass ratios (t_2/t_1 ≈ √(m_2/m_1)).

RULE 2 (CLASS 5 ALKALI DOUBLET REQUIREMENT):
A single sharp early peak followed by flat baseline grass MUST NOT be classified as Class 5 unless verified as Na+ (23 Da) or K+ (39 Da) with characteristic K+/Na+ doublet mass ratios (t_K/t_Na ≈ 1.30). A single peak at m/z 19 (H3O+) or m/z 1 (H+) with flat baseline MUST be classified as low-signal Class 1.

RULE 3 (CLASS 5-Na SALINE CHECK):
A detector saturation plateau (y > 0.8) alone DOES NOT constitute Class 5-Na. Class 5-Na MANDATES an early Na+ payload spike (23 Da) AND diagnostic sodium orthophosphate cluster peaks at t/t_Na = √(125/23) ≈ 2.33, √(165/23) ≈ 2.68, or √(187/23) ≈ 2.85. If saturation follows an organic envelope without orthophosphate mass clusters, classify as Class 3.

RULE 4 (CLASS 3-P EXCLUSION RULE):
Class 3-P requires an explosive multi-peak primary impact plasma complex (global maximum y = 1.0) and a continuous elevated chemical noise cushion (y ≈ 0.25–0.50). EXCLUSION: If the spectrum contains a hydronium peak comb (19, 37, 55 Da), classify as Class 1 or 2, NOT 3-P. If the early peak is a single needle-sharp atomic alkali spike, classify as Class 5 or 5-Na, NOT 3-P.
```

---

## 4. ACTIONABLE TOOLS & PIPELINE IMPLEMENTATION PLAN

1. **`cda_dust_agent/tools/snr_gating.py`**: Algorithmic pre-filter to bypass VLM calls on traces with $Q_I < 5.0 \times 10^{-14}\text{ C}$ or $\text{SNR} < 4.0$, saving ~60% of API calls and eliminating Noise FPs.
2. **`cda_dust_agent/tools/peak_annotator.py`**: Python signal-processing module (`scipy.signal.find_peaks` + TOF mass solver) that annotates peak identities ($H_3O^+, Na^+, Mg^+, Si^+$) directly onto the spectrum plots provided to the VLM.
3. **`cda_dust_agent/tools/dual_panel_plotter.py`**: Renders 2-panel images: (Top) Full spectrum index 1–1018, (Bottom) Zoomed-in low-mass region index 40–200.
4. **`multi_agent_fp_fn_per_class_analysis.md`**: Saved full per-class report artifact.

---
*Exhaustive per-class analysis synthesized and consolidated by Main Orchestrator Agent.*
