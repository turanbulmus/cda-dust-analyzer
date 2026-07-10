import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def main():
    df = pd.read_parquet("cda_dust_agent/data/raw/cda_train.parquet")
    df_chem = df[df["class"] != "Noise"]
    
    print("Listing properties of Chemical grains in training set with lowest range/peak metrics:")
    records = []
    for idx, row in df_chem.iterrows():
        spec = np.array(row["spectrum"])
        qi = float(row["qi_ampl"])
        
        smoothed = pd.Series(spec).rolling(window=15, center=True).mean().dropna().values
        smoothed_range = smoothed.max() - smoothed.min()
        
        inverted = 1.0 - spec
        peaks, _ = find_peaks(inverted, height=0.20, prominence=0.08, width=1)
        valid_peaks = [p for p in peaks if 20 < p < 620]
        
        records.append({
            "sclk": row["sclk"],
            "class": row["class"],
            "qi": qi,
            "range": smoothed_range,
            "peaks_count": len(valid_peaks)
        })
        
    df_res = pd.DataFrame(records)
    
    print("\nTop 15 Chemical Grains with Lowest Smoothed Range:")
    print(df_res.sort_values(by="range").head(15).to_string(index=False))
    
    print("\nChemical Grains with 0 Peaks:")
    print(df_res[df_res["peaks_count"] == 0].to_string(index=False))

if __name__ == "__main__":
    main()
