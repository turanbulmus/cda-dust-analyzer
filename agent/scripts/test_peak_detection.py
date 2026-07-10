import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def is_noise_by_peaks(spectrum, height_threshold=0.30, prominence_threshold=0.15, width_threshold=2):
    # Invert the signal: peaks go UP (baseline becomes low)
    inverted = 1.0 - np.array(spectrum)
    
    # Find peaks pointing UP in the inverted spectrum with width constraint
    peaks, properties = find_peaks(
        inverted, 
        height=height_threshold, 
        prominence=prominence_threshold,
        width=width_threshold
    )
    
    # Exclude early trigger artifacts (e.g. calibration pulses at index < 20 or index > 620)
    valid_peaks = [p for p in peaks if 20 < p < 620]
    
    # If no valid peaks, it is obvious noise
    return len(valid_peaks) == 0, len(valid_peaks)

def main():
    df = pd.read_parquet("cda_dust_agent/data/raw/cda_train.parquet")
    print(f"Loaded {len(df)} training samples for validation.")
    
    results = []
    for idx, row in df.iterrows():
        is_noise, n_peaks = is_noise_by_peaks(row["spectrum"])
        results.append({
            "sclk": row["sclk"],
            "true_class": row["class"],
            "is_noise_pred": is_noise,
            "n_peaks": n_peaks
        })
        
    df_res = pd.DataFrame(results)
    
    # Print metrics
    total_noise = len(df_res[df_res["true_class"] == "Noise"])
    detected_noise = len(df_res[(df_res["true_class"] == "Noise") & (df_res["is_noise_pred"] == True)])
    false_noise = len(df_res[(df_res["true_class"] != "Noise") & (df_res["is_noise_pred"] == True)])
    total_chemical = len(df_res[df_res["true_class"] != "Noise"])
    
    print("\n" + "="*50)
    print(" PEAK DETECTION PRE-FILTER EVALUATION (TRAINING SET)")
    print("="*50)
    print(f"Total True Noise: {total_noise}")
    print(f" -> Correctly Classified as Noise (True Negatives): {detected_noise} ({detected_noise/total_noise*100:.2f}%)")
    print(f" -> Misclassified as Chemical (False Positives for VLM): {total_noise - detected_noise} ({(total_noise - detected_noise)/total_noise*100:.2f}%)")
    print(f"\nTotal True Chemical: {total_chemical}")
    print(f" -> Correctly passed to VLM (True Positives for VLM): {total_chemical - false_noise} ({(total_chemical - false_noise)/total_chemical*100:.2f}%)")
    print(f" -> Misclassified as Noise (False Negatives - BYPASSED VLM): {false_noise} ({false_noise/total_chemical*100:.2f}%)")
    print("="*50)
    
    print("\nDetailed Peak info for True Noise samples:")
    noise_df = df_res[df_res["true_class"] == "Noise"]
    for idx, row in noise_df.iterrows():
        spec = np.array(df[df["sclk"] == row["sclk"]].iloc[0]["spectrum"])
        # Re-run to get peaks
        inverted = 1.0 - spec
        peaks, props = find_peaks(inverted, height=0.30, prominence=0.15, width=2)
        valid_peaks = [p for p in peaks if 20 < p < 620]
        peak_vals = [float(spec[p]) for p in valid_peaks]
        print(f"SCLK {row['sclk']}: Pred Noise={row['is_noise_pred']} | valid peaks count={len(valid_peaks)} | indices={valid_peaks} | min values={peak_vals}")

if __name__ == "__main__":
    main()
