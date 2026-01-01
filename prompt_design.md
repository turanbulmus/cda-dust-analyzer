# Prompt Design for CDA Class 4 Identification

**Model**: Gemini 3.0 Pro 
**Input**: Image of the Spectrum Plot
**Task**: Classification (Class 4 vs Other)

## Context
You are an expert Astrophysicist and Data Scientist specializing in Cosmic Dust Analyzer (CDA) data. You are tasked with identifying a specific class of dust impacts, labeled as "**Class 4**", from a dataset containing various signals and noise.

## Distinctive Features of Class 4
Based on statistical analysis of the training data, "Class 4" is characterized by:
1.  **Prominent Peak at Index ~640**: There is a statistically significant, sharp peak around index 642. This is the most defining feature.
2.  **Secondary Activity**: Minor peaks may distinguish it from pure noise, but the peak at ~640 is the primary indicator.
3.  **Contrast with Noise**: Noise typically lacks structured high-intensity peaks and appears as random low-amplitude fluctuations. "Class 1" may have different peak locations or shapes.

## Prompt
```text
You are an expert in analyzing cosmic dust spectra.
Your task is to determine if the provided spectrum plot represents a "Class 4" event.

### Analysis Criteria
1.  **Scan for a Key Feature**: Look specifically for a distinct, sharp vertical peak located approximately at 63% of the x-axis (around index 640 out of ~1024).
2.  **Evaluate Signal-to-Noise**: Compare the peak height to the baseline noise. Class 4 events have a clear signal sticking out above the noise floor at this specific location.
3.  **Ignore Irrelevant Noise**: Random low-amplitude jitter across the spectrum is expected background noise. Focus on structural peaks.

### Decision
-   **YES (Class 4)**: If the spectrum distinguishes itself with a clear peak around the 640 index mark.
-   **NO**: If the spectrum looks like random noise or has peaks in completely different locations (e.g., only at the beginning or end without the mid-range peak).

Please analyze the attached image and provide your classification.
Return your answer in the following JSON format:
{
  "is_class_4": boolean,
  "confidence": float (0.0 to 1.0),
  "reasoning": "brief explanation focused on the peak at index ~640"
}
```
