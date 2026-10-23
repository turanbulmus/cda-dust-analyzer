import io
import json
import re
import numpy as np
from scipy.ndimage import uniform_filter1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MultipleLocator

SAMPLE_RATE_MHZ = 100.0
SAMPLE_PERIOD_US = 1.0 / SAMPLE_RATE_MHZ
SPECTRUM_LENGTH = 1018
CROP_START, CROP_STOP = 10, 641
RENDER_DPI = 60

LABEL_3OH = "3-OH"
LABEL_NON_3OH = "non-3-OH"
BINARY_LABELS = [LABEL_3OH, LABEL_NON_3OH]


def sample_time_us(raw_sample_index):
    """Time of flight of a raw lvl2 sample in microseconds (index 0 = trigger)."""
    return np.asarray(raw_sample_index) * SAMPLE_PERIOD_US


def qm_scaling_lee_v2(spectrum, radius: int = 2) -> np.ndarray:
    """Applies shift, zero-replacement, log10, 1D Lee filter (radius=2), and [0, 1] min-max scaling."""
    spectrum = np.array(spectrum, dtype=float)
    min_val = np.min(spectrum)
    if min_val < 0:
        spectrum = spectrum + np.abs(min_val)

    positive_vals = spectrum[spectrum > 0]
    smallest_nonzero = np.min(positive_vals) if len(positive_vals) > 0 else 1e-12
    spectrum = np.where(spectrum <= 0, smallest_nonzero, spectrum)

    log_spec = np.log10(spectrum)

    window_size = 2 * radius + 1
    local_mean = uniform_filter1d(log_spec, size=window_size, mode="reflect")
    local_sqr_mean = uniform_filter1d(log_spec**2, size=window_size, mode="reflect")
    local_var = np.maximum(0, local_sqr_mean - local_mean**2)

    overall_noise_var = np.percentile(local_var, 25)
    if overall_noise_var == 0:
        overall_noise_var = np.mean(local_var) + 1e-12

    weights = local_var / (local_var + overall_noise_var)
    lee_smoothed = local_mean + weights * (log_spec - local_mean)

    spec_min = np.min(lee_smoothed)
    spec_max = np.max(lee_smoothed)
    range_val = spec_max - spec_min
    scaled_spectrum = (
        (lee_smoothed - spec_min) / range_val
        if range_val > 0
        else np.zeros_like(lee_smoothed)
    )
    return scaled_spectrum


def is_obvious_noise_3oh(spectrum) -> bool:
    """Stage-1 deterministic pre-filter for obvious flatline/low-variance noise spectra.

    Calibrated on all 18,325 labelled CDA spectra so that 0% of true 3-OH spectra are bypassed
    while ~24% of all spectra (33% of Noise) are bypassed directly to 'non-3-OH'.
    """
    arr = np.array(spectrum, dtype=float)
    if len(arr) == 0:
        return True
    std_val = float(np.std(arr))
    max_val = float(np.max(arr))
    mean_val = float(np.mean(arr))
    q99_val = float(np.percentile(arr, 99))

    if std_val < 0.08:
        return True
    if (max_val - mean_val) < 0.25:
        return True
    if (max_val - q99_val) < 0.05 and std_val < 0.22 and (max_val - mean_val) < 0.40:
        return True
    return False


def generate_spectrum_image_bytes(
    spectrum: np.ndarray, title: str = "Target Spectrum", dpi: int = RENDER_DPI
) -> bytes:
    """Renders a preprocessed 631-point CDA spectrum on the 0.0-6.4 us time-of-flight axis."""
    fig = Figure(figsize=(12, 6), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    t = sample_time_us(np.arange(CROP_START, CROP_START + len(spectrum)))
    ax.plot(t, spectrum, color="black", linewidth=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0.0, 6.4)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, which="major", linewidth=0.8)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.5)
    ax.set_xlabel("Time of flight (µs)", fontsize=13)
    ax.set_ylabel("log amplitude (scaled)", fontsize=13)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def normalise_binary_label(raw: str | None) -> str:
    """Normalises model output class string to '3-OH' or 'non-3-OH'."""
    if not isinstance(raw, str):
        return LABEL_NON_3OH
    label = raw.strip().strip("'\"")
    if label.lower().startswith("class "):
        label = label[6:].strip()
    key = label.lower().replace(" ", "").replace("_", "-")
    if key in ("3-oh", "3oh"):
        return LABEL_3OH
    return LABEL_NON_3OH


def parse_response(text: str) -> dict:
    """Parses JSON prediction response from Gemini and normalises the binary class label."""
    try:
        clean_text = re.sub(r"```json\s*|\s*```", "", text).strip()
        data = json.loads(clean_text)
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict):
            raw_cls = data.get("class") or data.get("class_label")
            data["class"] = normalise_binary_label(raw_cls)
            return data
    except Exception:
        pass
    return {"class": LABEL_NON_3OH, "explanation": text}
