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


# ---------------------------------------------------------------------------
# Step 1: Physics-Constrained Two-Anchor Mass-Domain Calibration & Telemetry
# ---------------------------------------------------------------------------
def calibrate_spectrum_mass_domain(
    spectrum: np.ndarray,
    a_min: float = 0.462,
    a_max: float = 0.476,
    u_min: float = 10.0,
    u_max: float = 200.0,
    u_step: float = 0.5,
) -> dict:
    """Performs physics-constrained two-anchor TOF-to-mass calibration using the
    Na+ (23.0 u) 90%-contour centroid and the sharp Na+(NaOH) (63.0 u) cluster peak.

    Returns a dictionary containing:
      - t_us: original time array (us)
      - u_raw: effective mass array (u) corresponding to t_us
      - u_grid: uniform mass grid [u_min..u_max] in steps of u_step
      - spec_grid: spectrum linearly interpolated onto u_grid
      - t23_us, t63_us, a_us, t0_us: solved calibration parameters
    """
    s = np.asarray(spectrum, dtype=float)
    n = len(s)
    t_us = sample_time_us(np.arange(CROP_START, CROP_START + n))

    # Locate Na+ (23 u) across both impact-triggered (1.4-2.4 us) and ion-triggered (0.6-2.5 us) windows
    search_lo, search_hi = min(50, n - 1), min(250, n)
    na_idx = search_lo + int(np.argmax(s[search_lo:search_hi]))
    # Compute 90%-contour centroid of Na+ peak to avoid split/flat-top trailing edge drift
    idx_arr = np.arange(n)
    na_mask = (
        (idx_arr >= max(0, na_idx - 15))
        & (idx_arr <= min(n - 1, na_idx + 15))
        & (s >= 0.90 * s[na_idx])
    )
    na_centroid_idx = float(np.mean(idx_arr[na_mask])) if np.any(na_mask) else float(na_idx)
    t23_us = (na_centroid_idx + CROP_START) * SAMPLE_PERIOD_US

    # Locate Na+(NaOH) (63 u) peak (+1.30 to +1.65 us after Na+ centroid)
    lo63 = max(0, int(round(na_centroid_idx)) + 130)
    hi63 = min(n, int(round(na_centroid_idx)) + 165)
    if lo63 < hi63:
        idx63 = lo63 + int(np.argmax(s[lo63:hi63]))
    else:
        idx63 = min(n - 1, int(round(na_centroid_idx)) + 148)
    t63_us = (idx63 + CROP_START) * SAMPLE_PERIOD_US

    denom = np.sqrt(63.0) - np.sqrt(23.0)
    raw_a = (t63_us - t23_us) / denom
    a_us = float(np.clip(raw_a, a_min, a_max))
    t0_us = float(0.5 * ((t23_us - a_us * np.sqrt(23.0)) + (t63_us - a_us * np.sqrt(63.0))))

    u_raw = np.where(t_us > t0_us, ((t_us - t0_us) / a_us) ** 2, 0.0)
    num_steps = int(round((u_max - u_min) / u_step)) + 1
    u_grid = np.linspace(u_min, u_max, num_steps)
    spec_grid = np.interp(u_grid, u_raw, s)

    return {
        "t_us": t_us,
        "u_raw": u_raw,
        "u_grid": u_grid,
        "spec_grid": spec_grid,
        "t23_us": round(float(t23_us), 3),
        "t63_us": round(float(t63_us), 3),
        "dt_63_23_us": round(float(t63_us - t23_us), 3),
        "a_us": round(a_us, 4),
        "t0_us": round(t0_us, 3),
    }


def extract_mass_telemetry_3oh(spectrum: np.ndarray, raw_spectrum: np.ndarray | None = None) -> dict:
    """Extracts deterministic 1D mass-calibrated features for 3-OH vs intra-Type-3 adjudication."""
    cal = calibrate_spectrum_mass_domain(spectrum)
    ug = cal["u_grid"]
    sg = cal["spec_grid"]

    def win_max(u_lo: float, u_hi: float) -> float:
        mask = (ug >= u_lo) & (ug <= u_hi)
        return float(np.max(sg[mask])) if np.any(mask) else 0.0

    def win_min(u_lo: float, u_hi: float) -> float:
        mask = (ug >= u_lo) & (ug <= u_hi)
        return float(np.min(sg[mask])) if np.any(mask) else 0.0

    def win_mean(u_lo: float, u_hi: float) -> float:
        mask = (ug >= u_lo) & (ug <= u_hi)
        return float(np.mean(sg[mask])) if np.any(mask) else 0.0

    def win_peak_u(u_lo: float, u_hi: float) -> float:
        mask = (ug >= u_lo) & (ug <= u_hi)
        if not np.any(mask):
            return 0.5 * (u_lo + u_hi)
        sub_u = ug[mask]
        sub_s = sg[mask]
        return float(sub_u[int(np.argmax(sub_s))])

    # Key peak heights & prominences on [0, 1] Lee-smoothed log scale
    h23 = win_max(21.0, 25.0)
    h63 = win_max(60.0, 66.0)
    prom_63 = h63 - 0.5 * (win_min(46.0, 56.0) + win_min(69.0, 77.0))
    h103 = win_max(100.0, 106.0)
    prom_103 = h103 - 0.5 * (win_min(89.0, 97.0) + win_min(110.0, 120.0))
    h143 = win_max(140.0, 146.0)
    prom_143 = h143 - win_min(112.0, 123.0)
    h183 = win_max(179.0, 186.0)

    # Diagnostic non-OH contaminants:
    # 1. Carbonate (3-Car): 129 u Na+(Na2CO3) excess over flanking [113, 121] u and [133, 138] u valleys
    base_129 = 0.5 * (win_mean(113.0, 121.0) + win_mean(133.0, 138.0))
    h129 = win_max(126.0, 132.0)
    excess_129 = h129 - base_129
    prom_129 = h129 - win_min(112.0, 123.0)
    pk_118_135_u = win_peak_u(118.0, 135.0)

    # 85 u Na+(Na2O) excess
    base_85 = 0.5 * (win_min(71.0, 79.0) + win_min(90.0, 97.0))
    excess_85 = win_max(83.0, 88.0) - base_85

    # 2. Chloride (3-Cl): 81 u Na+(NaCl) and 139 u Na+(NaCl)2
    prom_81 = win_max(78.5, 83.0) - win_min(69.0, 76.0)
    prom_139 = win_max(136.0, 141.0) - win_min(128.0, 135.0)

    # 3. Potassium / Hydrate (39/41 u) and 95/113 u
    h39_41 = win_max(38.0, 42.5)
    prom_39_41 = h39_41 - win_min(28.0, 35.0)
    prom_95 = win_max(93.0, 97.5) - win_min(86.0, 92.0)
    prom_113 = win_max(111.0, 115.5) - win_min(106.0, 110.5)

    # 4. Water-ice hydronium (19 u H3O+, 37 u, 55 u, 73 u)
    h19 = win_max(17.5, 20.5)
    prom_19 = h19 - win_min(14.0, 17.0)
    prom_55 = win_max(53.0, 57.0) - win_min(45.0, 51.0)
    prom_73 = win_max(71.0, 75.0) - win_min(66.0, 70.0)

    # 5. Phosphate (3-P): 125, 165, 187 u
    p125 = win_max(122.0, 127.5) - win_min(110.0, 120.0)
    p165 = win_max(161.0, 169.0) - 0.5 * (win_min(148.0, 159.0) + win_min(170.0, 179.0))
    p187 = win_max(183.0, 193.0) - win_min(170.0, 180.0)

    # Linear SNR(63 u) if raw unlogged spectrum is provided, else estimated from log prominence
    linear_snr_63 = None
    if raw_spectrum is not None:
        raw_arr = np.asarray(raw_spectrum, dtype=float)
        if len(raw_arr) == SPECTRUM_LENGTH:
            raw_arr = raw_arr[CROP_START:CROP_STOP]
        min_r = np.min(raw_arr)
        if min_r < 0:
            raw_arr = raw_arr + abs(min_r)
        raw_grid = np.interp(ug, cal["u_raw"], raw_arr)
        sig_63 = float(np.max(raw_grid[(ug >= 60.0) & (ug <= 66.0)]))
        bg_mask = ((ug >= 45.0) & (ug <= 56.0)) | ((ug >= 69.0) & (ug <= 77.0))
        bg_mean = float(np.mean(raw_grid[bg_mask]))
        bg_std = float(np.std(raw_grid[bg_mask])) + 1e-9
        linear_snr_63 = round((sig_63 - bg_mean) / bg_std, 1)

    # Physics veto checks for 3-OH:
    # 1. Standard Carbonate Veto (Rule A + B): 85.37% precision, 85.37% recall (F1=0.8537) on 18,305 archive
    passes_129_veto = bool(
        (excess_129 <= 0.036)
        and not ((prom_129 >= 0.060) and (128.0 <= pk_118_135_u <= 131.5))
    )
    # 2. Strict Neuro-Symbolic Veto (Rule A + B + C, adding Type-1 water-ice comb veto): 91.67% precision (3 FPs), 80.49% recall (F1=0.8571)
    passes_strict_veto = bool(
        passes_129_veto and not ((h19 >= 0.65) and (prom_55 >= 0.090))
    )

    return {
        "t23_us": cal["t23_us"],
        "t63_us": cal["t63_us"],
        "dt_63_23_us": cal["dt_63_23_us"],
        "a_us": cal["a_us"],
        "t0_us": cal["t0_us"],
        "h23": round(h23, 3),
        "h63": round(h63, 3),
        "prom_63": round(prom_63, 3),
        "linear_snr_63": linear_snr_63,
        "h103": round(h103, 3),
        "prom_103": round(prom_103, 3),
        "h143": round(h143, 3),
        "prom_143": round(prom_143, 3),
        "h183": round(h183, 3),
        "h129": round(h129, 3),
        "excess_129": round(excess_129, 4),
        "prom_129": round(prom_129, 3),
        "pk_118_135_u": round(pk_118_135_u, 1),
        "excess_85": round(excess_85, 3),
        "prom_81": round(prom_81, 3),
        "prom_139": round(prom_139, 3),
        "h39_41": round(h39_41, 3),
        "prom_39_41": round(prom_39_41, 3),
        "prom_95": round(prom_95, 3),
        "prom_113": round(prom_113, 3),
        "h19": round(h19, 3),
        "prom_19": round(prom_19, 3),
        "prom_55": round(prom_55, 3),
        "prom_73": round(prom_73, 3),
        "p125": round(p125, 3),
        "p165": round(p165, 3),
        "p187": round(p187, 3),
        "passes_129_veto": passes_129_veto,
        "passes_strict_veto": passes_strict_veto,
    }


def format_telemetry_table_3oh(tele: dict) -> str:
    """Formats the quantitative 1D mass-calibrated telemetry block for VLM consumption."""
    snr_str = f"Linear SNR(63u)={tele['linear_snr_63']}" if tele.get("linear_snr_63") is not None else f"LogProm(63u)={tele['prom_63']:.3f}"
    veto_flag = "PASS (<= 0.037, pure baseline)" if tele["passes_129_veto"] else "ALERT (> 0.037, elevated 129u carbonate/cluster signal)"
    return (
        f"[Two-Anchor Mass Calibration Telemetry]\n"
        f"- Anchors: Na+(23u)={tele['t23_us']:.2f} us (h={tele['h23']:.2f}), "
        f"Na+(NaOH)(63u)={tele['t63_us']:.2f} us (dt=+{tele['dt_63_23_us']:.2f} us, h={tele['h63']:.2f}, {snr_str}) | "
        f"Solved a={tele['a_us']:.4f} us/sqrt(u), t0={tele['t0_us']:.2f} us\n"
        f"- Hydroxide Comb: 103u h={tele['h103']:.2f} (prom={tele['prom_103']:.3f}) | "
        f"143u h={tele['h143']:.2f} (prom={tele['prom_143']:.3f}) | 183u h={tele['h183']:.2f}\n"
        f"- Carbonate (3-Car) Check: 129u excess_129={tele['excess_129']:.4f} [{veto_flag}], "
        f"prom_129={tele['prom_129']:.3f} (local max@{tele['pk_118_135_u']:.1f}u) | 85u excess={tele['excess_85']:.3f}\n"
        f"- Chloride (3-Cl) & Phosphate (3-P) Check: 81u prom={tele['prom_81']:.3f}, 139u prom={tele['prom_139']:.3f} | "
        f"125u={tele['p125']:.3f}, 165u={tele['p165']:.3f}, 187u={tele['p187']:.3f}\n"
        f"- Potassium (3-KNa) & Water-Ice (Type 1/2) Check: 39/41u h={tele['h39_41']:.2f} (prom={tele['prom_39_41']:.3f}), "
        f"95u prom={tele['prom_95']:.3f}, 113u prom={tele['prom_113']:.3f} | 19u h={tele['h19']:.2f}, 55u prom={tele['prom_55']:.3f}, 73u prom={tele['prom_73']:.3f}"
    )


def extract_mass_telemetry_3p(spectrum: np.ndarray, raw_spectrum: np.ndarray | None = None) -> dict:
    """Extracts deterministic 1D mass-calibrated features for 3-P orthophosphate triplet adjudication."""
    cal = calibrate_spectrum_mass_domain(spectrum)
    ug = cal["u_grid"]
    sg = cal["spec_grid"]

    def win_max(u_lo: float, u_hi: float) -> float:
        mask = (ug >= u_lo) & (ug <= u_hi)
        return float(np.max(sg[mask])) if np.any(mask) else 0.0

    def win_min(u_lo: float, u_hi: float) -> float:
        mask = (ug >= u_lo) & (ug <= u_hi)
        return float(np.min(sg[mask])) if np.any(mask) else 0.0

    def win_peak_u(u_lo: float, u_hi: float) -> float:
        mask = (ug >= u_lo) & (ug <= u_hi)
        if not np.any(mask):
            return 0.5 * (u_lo + u_hi)
        return float(ug[mask][int(np.argmax(sg[mask]))])

    h23 = win_max(21.0, 25.0)
    h63 = win_max(60.0, 66.0)
    prom_63 = h63 - 0.5 * (win_min(46.0, 56.0) + win_min(69.0, 77.0))

    p125 = win_max(122.0, 127.5) - win_min(110.0, 120.0)
    p129 = win_max(128.0, 133.0) - win_min(110.0, 120.0)
    diff_125_129 = win_max(122.0, 127.0) - win_max(128.5, 133.0)
    pk_118_135_u = win_peak_u(118.0, 135.0)

    p165 = win_max(161.0, 169.0) - 0.5 * (win_min(148.0, 159.0) + win_min(170.0, 179.0))
    pk_158_172_u = win_peak_u(158.0, 172.0)

    p187 = win_max(183.0, 193.0) - win_min(170.0, 180.0)
    pk_180_195_u = win_peak_u(180.0, 195.0)

    p81 = win_max(78.0, 84.0) - win_min(69.0, 76.0)
    ratio_81_63 = win_max(78.0, 84.0) / max(0.01, h63)
    p139 = win_max(136.0, 141.0) - win_min(128.0, 135.0)

    # Hydronium check (Type 1/2 water ice rejection)
    prom_37 = win_max(35.5, 38.5) - win_min(28.0, 34.0)
    prom_55 = win_max(53.0, 57.0) - win_min(45.0, 51.0)
    prom_73 = win_max(71.0, 75.0) - win_min(66.0, 70.0)

    # Tier-1 Canonical 3-P gate (derived strictly from catalogued Postberg et al. 2023 3-P grains, p81 <= 0.10)
    passes_canonical_3p_gate = bool(
        (diff_125_129 >= 0.0)
        and (p125 >= 0.050)
        and (p165 >= 0.070)
        and (p187 >= 0.100)
        and (p81 <= 0.10)
        and (p139 <= 0.145)
    )
    # Tier-2 Loose Physical Screen (parameter-free 125u >= 129u sign test + positive triplet, retains mixed-chloride SCLK 1511735208)
    passes_loose_3p_screen = bool(
        (h23 >= 0.70)
        and (h63 >= 0.50)
        and (prom_63 >= 0.10)
        and (diff_125_129 >= 0.0)
        and (p125 >= 0.040)
        and (p165 >= 0.040)
        and (p187 >= 0.040)
    )
    # Combined gate (retains mixed-chloride 3-P up to p81 <= 0.30)
    passes_3p_triplet_gate = bool(
        (diff_125_129 >= 0.0)
        and (p125 >= 0.050)
        and (p165 >= 0.045)
        and (p187 >= 0.065)
        and (p81 <= 0.30)
        and (ratio_81_63 <= 0.90)
        and (p139 <= 0.145)
    )
    triplet_score = round(float(p125 + p165 + p187 + diff_125_129 - 0.5 * max(0.0, p139)), 4)

    return {
        "t23_us": cal["t23_us"],
        "t63_us": cal["t63_us"],
        "a_us": cal["a_us"],
        "t0_us": cal["t0_us"],
        "h23": round(h23, 3),
        "h63": round(h63, 3),
        "prom_63": round(prom_63, 3),
        "p125": round(p125, 3),
        "p129": round(p129, 3),
        "diff_125_129": round(diff_125_129, 4),
        "pk_118_135_u": round(pk_118_135_u, 1),
        "p165": round(p165, 3),
        "pk_158_172_u": round(pk_158_172_u, 1),
        "p187": round(p187, 3),
        "pk_180_195_u": round(pk_180_195_u, 1),
        "p81": round(p81, 3),
        "ratio_81_63": round(ratio_81_63, 3),
        "p139": round(p139, 3),
        "prom_37": round(prom_37, 3),
        "prom_55": round(prom_55, 3),
        "prom_73": round(prom_73, 3),
        "passes_canonical_3p_gate": passes_canonical_3p_gate,
        "passes_loose_3p_screen": passes_loose_3p_screen,
        "passes_3p_triplet_gate": passes_3p_triplet_gate,
        "triplet_score": triplet_score,
    }


def format_telemetry_table_3p(tele: dict) -> str:
    """Formats raw measured 1D mass-calibrated quantities for 3-P adjudication without leaking PASS/FAIL flags."""
    return (
        f"[Two-Anchor Mass Calibration Telemetry for 3-P (Raw Measured Prominences)]\n"
        f"- Anchors: Na+(23u)={tele['t23_us']:.2f} us (h={tele['h23']:.2f}), "
        f"Na+(NaOH)(63u)={tele['t63_us']:.2f} us (h={tele['h63']:.2f}, prom={tele['prom_63']:.3f}) | "
        f"Solved a={tele['a_us']:.4f} us/sqrt(u), t0={tele['t0_us']:.2f} us\n"
        f"- 3-P Orthophosphate Triplet Measurements:\n"
        f"  * 125u Na+(NaPO3): prom={tele['p125']:.3f} vs 129u Carbonate prom={tele['p129']:.3f} "
        f"(diff_125_minus_129={tele['diff_125_129']:+.4f}, local apex@{tele['pk_118_135_u']:.1f}u)\n"
        f"  * 165u Na+(Na2HPO4): prom={tele['p165']:.3f} (local apex@{tele['pk_158_172_u']:.1f}u)\n"
        f"  * 187u Na+(Na3PO4): prom={tele['p187']:.3f} (local apex@{tele['pk_180_195_u']:.1f}u)\n"
        f"- Chloride (3-Cl) & Water-Ice Measurements: 81u prom={tele['p81']:.3f} (81u/63u height ratio={tele['ratio_81_63']:.2f}), "
        f"139u prom={tele['p139']:.3f} | Hydronium 37u={tele['prom_37']:.2f}, 55u={tele['prom_55']:.2f}, 73u={tele['prom_73']:.2f}"
    )


def generate_dual_panel_mass_spectrum_bytes(
    spectrum: np.ndarray,
    title: str = "Target Spectrum",
    mode: str = "3oh",
    dpi: int = 90,
) -> bytes:
    """Renders a 2-panel Physics-Calibrated Mass-Domain Figure:
      - Top Panel: Full calibrated mass spectrum (10 to 200 u) with Na+ and NaOH backbone lines.
      - Bottom Panel: Zoomed High-Mass Anion Diagnostic Window (70 to 195 u) with color-coded
        vertical mass lines for OH (103, 143, 183 u), Carbonate (85, 129, 169 u),
        Phosphate (125, 165, 187 u), Chloride (81, 139 u), and Potassium (95, 113 u).
    """
    cal = calibrate_spectrum_mass_domain(spectrum)
    ug = cal["u_grid"]
    sg = cal["spec_grid"]

    fig = Figure(figsize=(12, 7.5), dpi=dpi)
    FigureCanvasAgg(fig)

    # Top panel: Full Calibrated Mass Spectrum (10 - 200 u)
    ax1 = fig.add_subplot(211)
    ax1.plot(ug, sg, color="black", linewidth=1.5, zorder=3)
    ax1.set_title(
        f"{title} — Two-Anchor Mass Calibrated (Na+ 23u @ {cal['t23_us']:.2f} µs, "
        f"63u @ {cal['t63_us']:.2f} µs, a={cal['a_us']:.3f} µs/√u)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.set_xlim(10.0, 200.0)
    ax1.set_ylim(-0.03, 1.05)
    ax1.xaxis.set_major_locator(MultipleLocator(10.0))
    ax1.xaxis.set_minor_locator(MultipleLocator(2.0))
    ax1.grid(True, which="major", linewidth=0.6, alpha=0.5)
    ax1.grid(True, which="minor", linewidth=0.25, alpha=0.3)
    ax1.set_xlabel("Calibrated Effective Mass m/z (u) [Full Range 10–200 u]", fontsize=11)
    ax1.set_ylabel("Scaled Log Amplitude", fontsize=11)

    backbone_lines = [
        (19.0, "19 H3O+", "#008080"),
        (23.0, "23 Na+", "#1f77b4"),
        (39.0, "39 K+", "#9467bd"),
        (41.0, "41 Na(H2O)+", "#17becf"),
        (63.0, "63 Na(NaOH)+", "#1f77b4"),
        (103.0, "103 (NaOH)2", "#1f77b4"),
        (143.0, "143 (NaOH)3", "#1f77b4"),
        (183.0, "183 (NaOH)4", "#1f77b4"),
    ]
    for m_u, lbl, col in backbone_lines:
        ax1.axvline(m_u, color=col, linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)
        ax1.text(
            m_u + 0.6,
            0.94,
            lbl,
            color=col,
            fontsize=8.5,
            rotation=90,
            va="top",
            ha="left",
            fontweight="bold",
        )

    # Bottom panel: Zoomed High-Mass Anion Diagnostic Window (70 - 195 u)
    ax2 = fig.add_subplot(212)
    mask_zoom = (ug >= 70.0) & (ug <= 195.0)
    uz = ug[mask_zoom]
    sz = sg[mask_zoom]
    ax2.plot(uz, sz, color="black", linewidth=1.8, zorder=4)

    y_lo = max(-0.02, float(np.min(sz)) - 0.05) if len(sz) else -0.02
    y_hi = min(1.05, float(np.max(sz)) + 0.08) if len(sz) else 1.02
    if y_hi - y_lo < 0.25:
        y_hi = min(1.05, y_lo + 0.25)
    ax2.set_xlim(70.0, 195.0)
    ax2.set_ylim(y_lo, y_hi)
    ax2.xaxis.set_major_locator(MultipleLocator(5.0))
    ax2.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax2.grid(True, which="major", linewidth=0.6, alpha=0.5)
    ax2.grid(True, which="minor", linewidth=0.3, alpha=0.3)
    ax2.set_xlabel("Calibrated Effective Mass m/z (u) [Zoomed Anion Diagnostic Window 70–195 u]", fontsize=11)
    ax2.set_ylabel("Scaled Log Amplitude (Zoom)", fontsize=11)

    # Shaded diagnostic bands for 125u (3-P) vs 129u (3-Car)
    ax2.axvspan(123.0, 127.0, color="#2ca02c", alpha=0.12, label="125u 3-P Band")
    ax2.axvspan(127.5, 131.5, color="#d62728", alpha=0.14, label="129u 3-Car Band")
    if mode == "3p":
        ax2.axvspan(162.0, 168.0, color="#2ca02c", alpha=0.12)
        ax2.axvspan(184.0, 190.0, color="#2ca02c", alpha=0.12)

    anion_lines = [
        (81.0, "81 Cl/Hyd", "#ff7f0e", ":"),
        (85.0, "85 Car(Na2O)", "#d62728", "--"),
        (95.0, "95 KNa", "#9467bd", ":"),
        (103.0, "103 OH_2", "#1f77b4", "-."),
        (113.0, "113 KNa", "#9467bd", ":"),
        (125.0, "125 3-P", "#2ca02c", "--"),
        (129.0, "129 3-Car", "#d62728", "--"),
        (139.0, "139 3-Cl", "#ff7f0e", ":"),
        (143.0, "143 OH_3", "#1f77b4", "-."),
        (165.0, "165 3-P", "#2ca02c", "--"),
        (169.0, "169 3-Car", "#d62728", ":"),
        (183.0, "183 OH_4", "#1f77b4", "-."),
        (187.0, "187 3-P", "#2ca02c", "--"),
    ]
    y_top = y_hi - 0.02 * (y_hi - y_lo)
    for m_u, lbl, col, ls in anion_lines:
        ax2.axvline(m_u, color=col, linestyle=ls, linewidth=1.2, alpha=0.85, zorder=3)
        ax2.text(
            m_u + 0.35,
            y_top,
            lbl,
            color=col,
            fontsize=8.5,
            rotation=90,
            va="top",
            ha="left",
            fontweight="bold",
        )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


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
