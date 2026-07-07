import pandas as pd
import numpy as np

def main():
    df = pd.read_parquet("cda_dust_agent/data/raw/cda_train.parquet")
    
    # Class 1
    c1 = df[df["class"] == "1"].head(2)
    print("Class 1 Minima Locations:")
    for idx, row in c1.iterrows():
        spec = np.array(row["spectrum"])
        # Find local minima
        minima = []
        for i in range(1, len(spec)-1):
            if spec[i] < spec[i-1] and spec[i] < spec[i+1] and spec[i] < 0.7:
                minima.append((i, float(spec[i])))
        print(f"SCLK {row['sclk']}: minima count={len(minima)}, first 5 minima index/value={minima[:5]}")

    # Class 4
    c4 = df[df["class"] == "4"].head(2)
    print("\nClass 4 Minima Locations:")
    for idx, row in c4.iterrows():
        spec = np.array(row["spectrum"])
        minima = []
        for i in range(1, len(spec)-1):
            if spec[i] < spec[i-1] and spec[i] < spec[i+1] and spec[i] < 0.7:
                minima.append((i, float(spec[i])))
        print(f"SCLK {row['sclk']}: minima count={len(minima)}, first 5 minima index/value={minima[:5]}")

if __name__ == "__main__":
    main()
