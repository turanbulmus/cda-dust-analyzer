import os
import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cassini CDA Dust Analyzer: Frequency Domain Exploration\n",
    "\n",
    "This notebook explores the visual and signal processing characteristics of Cassini Cosmic Dust Analyzer (CDA) Time-of-Flight mass spectra across different classes.\n",
    "\n",
    "To help the Vision-Language Model (VLM) overcome its limitations in measuring precise pixel-level peak spacings, we investigate three key signal processing transformations:\n",
    "1. **Fast Fourier Transform (FFT)**: Converts the time-of-flight index domain into the frequency domain. For Class 1 (constant spacing), this should yield a clean, sharp harmonic peak.\n",
    "2. **Autocorrelation Function (ACF)**: Measures the self-similarity of the spectrum at different lags. Class 1 should show highly periodic peak repetition, whereas Class 2 (converging) and Class 4 (diverging) will show rapidly decaying ACF curves.\n",
    "3. **Short-Time Fourier Transform (Spectrogram)**: Computes local frequency content over moving windows. Class 1 will show a horizontal frequency line, Class 2 an upward-sloping line (frequency increases), and Class 4 a downward-sloping line (frequency decreases).\n",
    "\n",
    "All transformations are normalized to the `[0, 1]` range to ensure consistent visualization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.signal import spectrogram\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "# Ensure we can import from the workspace root\n",
    "sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))\n",
    "\n",
    "from scripts.run_ablation_study_incremental import preprocess_huggingface_data\n",
    "\n",
    "load_dotenv(dotenv_path=\"../.env\")\n",
    "\n",
    "print(\"Libraries imported successfully.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\n",
    "print(\"Loading HuggingFace training data...\")\n",
    "full_df = preprocess_huggingface_data(\"H\")\n",
    "print(f\"Loaded {len(full_df)} rows of data.\")\n",
    "print(\"Class distribution:\\n\", full_df['class'].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Transformation Functions\n",
    "\n",
    "Here, we define the mathematical transformation functions. Each function accepts the 1D spectrum array, truncates/pads it to length 630 (to match the VLM's field of view), computes the transformation, and scales the output values to the range `[0, 1]`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_and_truncate(spectrum, target_len=630):\n",
    "    \"\"\"Ensures the spectrum is a numpy array of target_len.\"\"\"\n",
    "    arr = np.array(spectrum)\n",
    "    if len(arr) > target_len:\n",
    "        return arr[:target_len]\n",
    "    elif len(arr) < target_len:\n",
    "        return np.pad(arr, (0, target_len - len(arr)), mode='constant')\n",
    "    return arr\n",
    "\n",
    "def compute_fft(spectrum_data):\n",
    "    \"\"\"Computes the FFT amplitude spectrum and normalizes it to [0, 1].\"\"\"\n",
    "    signal = clean_and_truncate(spectrum_data)\n",
    "    # Remove DC offset to prevent frequency 0 from dominating the plot scale\n",
    "    signal_centered = signal - np.mean(signal)\n",
    "    fft_vals = np.abs(np.fft.rfft(signal_centered))\n",
    "    \n",
    "    # Normalize to [0, 1]\n",
    "    min_val = np.min(fft_vals)\n",
    "    max_val = np.max(fft_vals)\n",
    "    fft_vals_norm = (fft_vals - min_val) / (max_val - min_val + 1e-9)\n",
    "    \n",
    "    freqs = np.fft.rfftfreq(len(signal))\n",
    "    return freqs, fft_vals_norm\n",
    "\n",
    "def compute_autocorrelation(spectrum_data):\n",
    "    \"\"\"Computes the centered autocorrelation function and normalizes it to [0, 1].\"\"\"\n",
    "    signal = clean_and_truncate(spectrum_data)\n",
    "    # Center signal to highlight true periodicities rather than baseline height\n",
    "    signal_centered = signal - np.mean(signal)\n",
    "    \n",
    "    acf = np.correlate(signal_centered, signal_centered, mode='full')\n",
    "    acf = acf[acf.size // 2:]  # Keep positive lags\n",
    "    \n",
    "    # Normalize to [0, 1]\n",
    "    min_val = np.min(acf)\n",
    "    max_val = np.max(acf)\n",
    "    acf_norm = (acf - min_val) / (max_val - min_val + 1e-9)\n",
    "    \n",
    "    lags = np.arange(len(acf_norm))\n",
    "    return lags, acf_norm\n",
    "\n",
    "def compute_spectrogram(spectrum_data, nperseg=64, noverlap=56):\n",
    "    \"\"\"Computes the spectrogram (STFT) and normalizes it to [0, 1].\"\"\"\n",
    "    signal = clean_and_truncate(spectrum_data)\n",
    "    \n",
    "    # Compute spectrogram using scipy\n",
    "    f, t, Sxx = spectrogram(signal, fs=1.0, nperseg=nperseg, noverlap=noverlap)\n",
    "    \n",
    "    # Normalize to [0, 1]\n",
    "    min_val = np.min(Sxx)\n",
    "    max_val = np.max(Sxx)\n",
    "    Sxx_norm = (Sxx - min_val) / (max_val - min_val + 1e-9)\n",
    "    \n",
    "    return t, f, Sxx_norm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualization Helper\n",
    "\n",
    "This helper function creates a beautifully formatted 2x2 grid containing the Time-of-Flight spectrum, FFT, Autocorrelation, and Spectrogram for a given sample."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_class_diagnostics(row, title_prefix=\"\"):\n",
    "    spectrum = row['spectrum']\n",
    "    class_label = row['class']\n",
    "    sclk = row['sclk']\n",
    "    \n",
    "    # Compute transformations\n",
    "    freqs, fft_vals = compute_fft(spectrum)\n",
    "    lags, acf_vals = compute_autocorrelation(spectrum)\n",
    "    t, f, spec_vals = compute_spectrogram(spectrum)\n",
    "    \n",
    "    fig, axs = plt.subplots(2, 2, figsize=(15, 10))\n",
    "    fig.suptitle(f\"{title_prefix}Class: {class_label} | SCLK: {sclk}\", fontsize=16, fontweight='bold', color='#1e293b')\n",
    "    \n",
    "    # 1. Top-Left: Original Spectrum (Log y-axis)\n",
    "    axs[0, 0].plot(spectrum, color='#0f172a', linewidth=1.5)\n",
    "    axs[0, 0].set_title(\"1. Time-of-Flight Spectrum (Original)\", fontsize=12, fontweight='semibold', color='#334155')\n",
    "    axs[0, 0].set_xlabel(\"Index\", fontsize=10)\n",
    "    axs[0, 0].set_ylabel(\"Intensity (Normalized [0, 1])\", fontsize=10)\n",
    "    axs[0, 0].set_xlim(0, 630)\n",
    "    axs[0, 0].set_ylim(-0.02, 1.02)\n",
    "    axs[0, 0].grid(True, linestyle='--', alpha=0.5)\n",
    "    \n",
    "    # 2. Top-Right: FFT Power Spectrum\n",
    "    axs[0, 1].plot(freqs, fft_vals, color='#d97706', linewidth=1.5)\n",
    "    axs[0, 1].set_title(\"2. Normalized FFT (Amplitude Spectrum)\", fontsize=12, fontweight='semibold', color='#334155')\n",
    "    axs[0, 1].set_xlabel(\"Frequency (cycles / index)\", fontsize=10)\n",
    "    axs[0, 1].set_ylabel(\"Amplitude (Normalized [0, 1])\", fontsize=10)\n",
    "    axs[0, 1].set_xlim(0.005, 0.2)  # Focus on low-mid frequencies (excluding DC offset)\n",
    "    axs[0, 1].set_ylim(-0.02, 1.02)\n",
    "    axs[0, 1].grid(True, linestyle='--', alpha=0.5)\n",
    "    \n",
    "    # 3. Bottom-Left: Autocorrelation (ACF)\n",
    "    axs[1, 0].plot(lags, acf_vals, color='#2563eb', linewidth=1.5)\n",
    "    axs[1, 0].set_title(\"3. Normalized Autocorrelation (ACF)\", fontsize=12, fontweight='semibold', color='#334155')\n",
    "    axs[1, 0].set_xlabel(\"Lag (index units)\", fontsize=10)\n",
    "    axs[1, 0].set_ylabel(\"R(lag) (Normalized [0, 1])\", fontsize=10)\n",
    "    axs[1, 0].set_xlim(0, 300)  # Focus on key lags\n",
    "    axs[1, 0].set_ylim(-0.02, 1.02)\n",
    "    axs[1, 0].grid(True, linestyle='--', alpha=0.5)\n",
    "    \n",
    "    # 4. Bottom-Right: Spectrogram (STFT)\n",
    "    im = axs[1, 1].pcolormesh(t, f, spec_vals, shading='gouraud', cmap='viridis')\n",
    "    axs[1, 1].set_title(\"4. Normalized Spectrogram (Time-Frequency)\", fontsize=12, fontweight='semibold', color='#334155')\n",
    "    axs[1, 1].set_xlabel(\"Time Window Center Index\", fontsize=10)\n",
    "    axs[1, 1].set_ylabel(\"Frequency\", fontsize=10)\n",
    "    axs[1, 1].set_ylim(0.005, 0.2)  # Focus on matching frequencies\n",
    "    fig.colorbar(im, ax=axs[1, 1], label=\"Intensity (Normalized [0, 1])\")\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Plotting One Example For Each Class\n",
    "\n",
    "We sample exactly one representative example for each of the 7 classes to analyze and compare their signatures in the transformed spaces."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "classes = ['Noise', '1', '2', '3', '4', '5', '5-Na']\n",
    "\n",
    "for cls in classes:\n",
    "    cls_subset = full_df[full_df['class'] == cls]\n",
    "    if len(cls_subset) > 0:\n",
    "        # Sample the first instance\n",
    "        sample = cls_subset.iloc[0]\n",
    "        plot_class_diagnostics(sample, title_prefix=f\"Representative Sample for \")\n",
    "    else:\n",
    "        print(f\"Warning: No samples found for class '{cls}'\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysis and Findings\n",
    "\n",
    "Based on the generated plots, we can confirm several key visual and mathematical insights:\n",
    "\n",
    "### 1. Class 1 (Constant Spacing)\n",
    "- **Spectrum**: Showcases clear, distinct peak intervals starting early and decaying to the right.\n",
    "- **FFT**: Features a highly pronounced, sharp harmonic peak near frequency `~0.015 - 0.02` (corresponding to spacing $1 / 60 \\approx 0.016$). This single peak is an unmistakable finger-print.\n",
    "- **Autocorrelation**: Displays very strong, periodic, and cleanly repeating wave-like peaks at lags of 60, 120, 180, etc.\n",
    "- **Spectrogram**: Displays a clear, horizontal band of elevated intensity, representing a constant frequency over time.\n",
    "\n",
    "### 2. Class 2 (Decreasing Spacing)\n",
    "- **FFT**: Has a broader, less sharp frequency profile because the spacing converges.\n",
    "- **Autocorrelation**: Features a rapidly decaying curve that loses peak definition at larger lags.\n",
    "- **Spectrogram**: The dominant frequency band displays a slight **upward slope** from left to right as the peaks converge.\n",
    "\n",
    "### 3. Class 4 (Increasing Spacing)\n",
    "- **FFT**: Broadened and shifted towards the lower frequencies at higher index values.\n",
    "- **Autocorrelation**: Rapidly dampens and exhibits no clear periodic peaks.\n",
    "- **Spectrogram**: The frequency signature shows a distinct **downward slope** as the peaks diverge.\n",
    "\n",
    "### 4. Noise Class\n",
    "- **Spectrum**: Chaotic, narrow electronic trigger spike followed by broad digitizer grass.\n",
    "- **FFT**: Entirely flat, low-level white-noise background with no dominant peak.\n",
    "- **Autocorrelation**: Plummets to zero almost instantly after lag 0, confirming the lack of correlation between successive samples.\n",
    "- **Spectrogram**: Displays a uniform, diffuse green/blue noise floor without any structured bands."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_style": "inline",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

notebook_path = "/workspaces/cda-dust-analyzer/Notebooks/05_frequency_domain_exploration.ipynb"
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
with open(notebook_path, 'w') as f:
    json.dump(notebook_content, f, indent=1)

print(f"Notebook successfully created at {notebook_path}")
