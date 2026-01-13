import pandas as pd
import matplotlib.pyplot as plt
import os

# IDs provided by the user
TARGET_IDS = [
    1503251881, 1569847688, 1501671551, 1573982459, 
    1495340170, 1489089036, 1514172190, 1530427866, 
    1519606812, 1530422688, 1530416091, 1530429162
]

def generate_plot(row, output_dir):
    sclk = row['sclk']
    cls = str(row['class'])
    spectrum = row['spectrum']
    
    plt.figure(figsize=(12, 6))
    # Log scale as per previous requirements
    plt.semilogy(spectrum, color='black', linewidth=1.5)
    
    # Highlight regions
    plt.axvspan(180, 400, color='green', alpha=0.1, label='Class 4 Region')
    plt.axvspan(0, 50, color='red', alpha=0.1, label='Noise Region')
    
    # Title with Classification
    title = f"ID: {sclk} | Class: {cls}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Save
    filename = f"{sclk}_class_{cls}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved {filepath}")

def main():
    data_path = 'data/cda_sample.parquet'
    output_dir = 'data'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Filter for target IDs
    # Ensure sclk is int or str matching TARGET_IDS
    # Checking dtype first might be good, but let's assume int first or convert
    
    # Convert TARGET_IDS to same type as df['sclk'] if needed
    # Let's check a sample from df first or just try both
    
    # Filter
    subset = df[df['sclk'].isin(TARGET_IDS)]
    
    if subset.empty:
        print("No matches found for provided IDs.")
        # Try string conversion if empty (just in case sclk is str in parquet)
        subset_str = df[df['sclk'].astype(str).isin([str(x) for x in TARGET_IDS])]
        if not subset_str.empty:
            subset = subset_str
            print("Found matches using string comparison.")
        else:
            print("Still no matches. Check IDs.")
            return

    print(f"Found {len(subset)} matching records.")
    
    for _, row in subset.iterrows():
        generate_plot(row, output_dir)

if __name__ == "__main__":
    main()
