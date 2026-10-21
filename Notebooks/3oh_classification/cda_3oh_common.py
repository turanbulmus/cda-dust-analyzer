"""Shared code for the 3-OH vs non-3-OH study.

Used by 15_3oh_few_shot_study.ipynb (few-shot pool + labelled evaluation) and
16_3oh_inference.ipynb (unlabelled data). It follows the 3-P peaks-focused pipeline
(Notebooks/XX_3p_ablation_study_peaks_focused.ipynb, Notebooks/XX_peaks_focused_inference.ipynb):
same crop, Q_I window, Lee v2 scaling, Vertex AI batch jobs and JSON response format.

Differences from the 3-P code:
  * plots use a time-of-flight axis in microseconds instead of the sample index;
  * the few-shot pool (exemplar images, explanations, prompts) is written once by notebook 15
    and reloaded by notebook 16, so inference uses exactly the exemplars that were evaluated;
  * the SCLK of each prediction is read back from the echoed request, not from the model's answer.
"""
import base64
import io
import json
import os
import re
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import uniform_filter1d
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Instrument time base
# ---------------------------------------------------------------------------
# CDA Chemical Analyser: the multiplier signal is log-amplified, digitised at 8 bit and sampled
# at 100 MHz (10 ns per sample) for 6.4 us after the trigger; an extended spectrum follows at
# 10 MHz (Postberg et al. 2008 Sec. 2; 2018 and 2023 Methods; CDA Data Handbook Sec. 1.1.2).
# The 0.375 MHz rate in the Data Handbook is the pre-trigger rate of the QC/QT/QI charge
# channels, not of the mass spectrum.
SAMPLE_RATE_MHZ = 100.0
SAMPLE_PERIOD_US = 1.0 / SAMPLE_RATE_MHZ
SPECTRUM_LENGTH = 1018
# Raw lvl2 samples kept: 10..640, i.e. 0.10..6.40 us = the 100 MHz high-rate window.
CROP_START, CROP_STOP = 10, 641
# t = t0 + a * sqrt(m/z) for singly charged ions (Postberg et al. 2023, Methods).
STRETCH_A_US = 0.473
# Same Q_I window as the 3-P study: 10 fC <= Q_I <= 2 pC.
QI_MIN_C, QI_MAX_C = 1e-14, 2e-12
RENDER_DPI = 60

HF_REPO_ID = "CosmicDustGroup/cassini-cda-spectra"
HF_TRAIN_FILE = "data/lvl2/cda_qm_spectra_pre2008277_train_lvl2.parquet"
HF_INF_FILE = "data/lvl2/cda_qm_spectra_pre2008277_inf_lvl2.parquet"
META_COLUMNS = ["sclk", "event_id", "class", "qi_ampl", "spectrum_length"]

LABEL_3OH = "3-OH"
LABEL_NON_3OH = "non-3-OH"
BINARY_LABELS = [LABEL_3OH, LABEL_NON_3OH]


def time_base_settings():
    """Settings that must match between the few-shot pool and the inference run."""
    return {
        "sample_rate_mhz": SAMPLE_RATE_MHZ,
        "crop": [CROP_START, CROP_STOP],
        "stretch_a_us": STRETCH_A_US,
        "qi_window_c": [QI_MIN_C, QI_MAX_C],
        "render_dpi": RENDER_DPI,
    }


def sample_time_us(raw_sample_index):
    """Time of flight of a raw lvl2 sample (index 0 = trigger)."""
    return np.asarray(raw_sample_index) * SAMPLE_PERIOD_US


def delta_t_us(mass_from, mass_to, a_us=STRETCH_A_US):
    """Flight-time difference between two singly charged ions; independent of t0."""
    return a_us * (np.sqrt(mass_to) - np.sqrt(mass_from))


def mass_line_table(mass_lines, anchor_mass=23.0, reference_mass=63.0):
    """Translate (mass, ion, role) entries into time distances in us and samples."""
    rows = []
    for mass, ion, role in mass_lines:
        dt_anchor = float(delta_t_us(anchor_mass, mass))
        rows.append({
            "m/z (u)": mass,
            "ion": ion,
            "role": role,
            f"dt after m/z {anchor_mass:g} (us)": round(dt_anchor, 3),
            f"dt after m/z {anchor_mass:g} (samples)": int(round(dt_anchor / SAMPLE_PERIOD_US)),
            f"dt after m/z {reference_mass:g} (us)": round(float(delta_t_us(reference_mass, mass)), 3),
        })
    return pd.DataFrame(rows)


def locate_mass_lines(spectrum_cropped, masses, anchor_mass=23.0, search_halfwidth=8):
    """Take the highest peak as the anchor ion, derive t0 from it, and look for each mass line
    within +-search_halfwidth samples of its expected time. Returns (t0_us, rows)."""
    s = np.asarray(spectrum_cropped, dtype=float)
    i_anchor = int(np.argmax(s))
    t0 = float(sample_time_us(i_anchor + CROP_START)) - STRETCH_A_US * np.sqrt(anchor_mass)
    rows = []
    for mass in masses:
        t_expected = t0 + STRETCH_A_US * np.sqrt(mass)
        i_expected = int(round(t_expected / SAMPLE_PERIOD_US)) - CROP_START
        lo, hi = max(i_expected - search_halfwidth, 0), min(i_expected + search_halfwidth + 1, len(s))
        if lo >= hi:
            rows.append({"m/z": mass, "t_expected_us": round(t_expected, 2), "t_observed_us": None, "height": None})
            continue
        i_observed = lo + int(np.argmax(s[lo:hi]))
        rows.append({
            "m/z": mass,
            "t_expected_us": round(t_expected, 2),
            "t_observed_us": round(float(sample_time_us(i_observed + CROP_START)), 2),
            "height": round(float(s[i_observed]), 2),
        })
    return t0, rows


# ---------------------------------------------------------------------------
# Data loading and preprocessing (identical to the 3-P study)
# ---------------------------------------------------------------------------
def hf_path(filename):
    import huggingface_hub
    return huggingface_hub.hf_hub_download(repo_id=HF_REPO_ID, filename=filename, repo_type="dataset")


def read_metadata(path):
    """All rows of a lvl2 parquet file without the spectra."""
    return pq.read_table(path, columns=META_COLUMNS).to_pandas()


def read_spectra(path, sclks, chunk_rows=50_000):
    """Rows (with raw spectra) whose SCLK is in `sclks`. The inference file is one 600 MB row
    group, so it is streamed in chunks instead of being decoded in one piece."""
    wanted = pa.array(sorted({int(s) for s in sclks}), type=pa.int64())
    columns = META_COLUMNS + ["spectrum"]
    if len(wanted) == 0:
        return pd.DataFrame(columns=columns)
    pieces = []
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=chunk_rows, columns=columns):
        mask = pc.is_in(batch.column("sclk"), value_set=wanted)
        if pc.any(mask).as_py():
            pieces.append(batch.filter(mask))
    if not pieces:
        return pd.DataFrame(columns=columns)
    return pa.Table.from_batches(pieces).to_pandas()


def in_qi_window(df):
    return df[(df["qi_ampl"] >= QI_MIN_C) & (df["qi_ampl"] <= QI_MAX_C)]


def qm_scaling_lee_v2(spectrum, radius=2):
    spectrum = np.array(spectrum, dtype=float)

    # Step 1: If min < 0, shift spectrum by abs(min) so smallest value becomes 0
    min_val = np.min(spectrum)
    if min_val < 0:
        spectrum = spectrum + np.abs(min_val)

    # Step 2: Replace 0 with the smallest non-zero positive number in the spectrum
    positive_vals = spectrum[spectrum > 0]
    smallest_nonzero = np.min(positive_vals) if len(positive_vals) > 0 else 1e-12
    spectrum = np.where(spectrum <= 0, smallest_nonzero, spectrum)

    # Step 3: Take log10
    log_spec = np.log10(spectrum)

    # Step 4: 1D Lee Filter (n=2 neighbors -> window_size = 2*2 + 1 = 5)
    window_size = 2 * radius + 1
    local_mean = uniform_filter1d(log_spec, size=window_size, mode='reflect')
    local_sqr_mean = uniform_filter1d(log_spec**2, size=window_size, mode='reflect')
    local_var = np.maximum(0, local_sqr_mean - local_mean**2)

    overall_noise_var = np.percentile(local_var, 25)
    if overall_noise_var == 0:
        overall_noise_var = np.mean(local_var) + 1e-12

    weights = local_var / (local_var + overall_noise_var)
    lee_smoothed = local_mean + weights * (log_spec - local_mean)

    # Step 5: Min-max scale to [0, 1]
    spec_min = np.min(lee_smoothed)
    spec_max = np.max(lee_smoothed)
    range_val = spec_max - spec_min
    scaled_spectrum = (lee_smoothed - spec_min) / range_val if range_val > 0 else np.zeros_like(lee_smoothed)
    return scaled_spectrum


def preprocess(df):
    """Length filter, crop to 0.10-6.40 us and Lee v2 scaling, as in the 3-P study."""
    df = df[df["spectrum_length"] == SPECTRUM_LENGTH].copy()
    df["spectrum"] = df["spectrum"].apply(
        lambda s: qm_scaling_lee_v2(np.asarray(s, dtype=float)[CROP_START:CROP_STOP])
    )
    return df


def load_catalogue(csv_path):
    """Expert catalogue (Notebooks/cda_class_list.csv) as sclk -> class."""
    cat = pd.read_csv(csv_path, dtype=str)
    cat["sclk"] = pd.to_numeric(cat["sclk"], errors="coerce")
    cat = cat.dropna(subset=["sclk"]).drop_duplicates("sclk")
    cat["sclk"] = cat["sclk"].astype("int64")
    return cat[["sclk", "class", "event_id"]].rename(columns={"class": "catalogue_class", "event_id": "catalogue_event_id"})


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_spectrum_png(spectrum_cropped, title, dpi=RENDER_DPI):
    """PNG of a preprocessed spectrum on a time-of-flight axis (0-6.4 us), 720x360 px at 60 dpi."""
    fig = Figure(figsize=(12, 6), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    t = sample_time_us(np.arange(CROP_START, CROP_START + len(spectrum_cropped)))
    ax.plot(t, spectrum_cropped, color="black", linewidth=1.5)
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


# ---------------------------------------------------------------------------
# Batch requests
# ---------------------------------------------------------------------------
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "id": {"type": "STRING", "description": "The provided sclk id"},
        "class": {"type": "STRING", "enum": BINARY_LABELS, "description": "The predicted label"},
        "explanation": {"type": "STRING", "description": "Brief explanation based on peak times"},
        "k": {"type": "INTEGER", "description": "The number of reference examples shown"},
    },
    "required": ["id", "class", "explanation", "k"],
}


def build_few_shot_parts(pool):
    parts = [{"text": pool["classification_prompt"]}, {"text": "Here are reference examples:"}]
    for ex in pool["exemplars"]:
        profile = pool["profiles"][ex["label"]]
        parts.append({"text": f"Example: {ex['label']}\n- General Profile: {profile}\n- Specific Sample Features: {ex['explanation']}"})
        parts.append({"inline_data": {"mime_type": "image/png", "data": ex["image_base64"]}})
    parts.append({"text": "Now, analyze the following spectrum:"})
    return parts


def build_request_line(few_shot_parts, system_instruction, sclk, image_b64, k, temperature=None):
    generation_config = {"responseMimeType": "application/json", "responseSchema": RESPONSE_SCHEMA}
    if temperature is not None:
        generation_config["temperature"] = temperature
    parts = few_shot_parts + [
        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
        {"text": f"Sample ID (sclk): {sclk} | k: {k}"},
    ]
    return json.dumps({
        "request": {
            "contents": [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": generation_config,
        }
    })


def write_request_file(path, df, pool):
    """One request per row of `df` (columns sclk, preprocessed spectrum) with the pool's exemplars."""
    few_shot_parts = build_few_shot_parts(pool)
    k = len(pool["exemplars"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Rendering requests"):
            image_b64 = base64.b64encode(render_spectrum_png(row.spectrum, f"Sample {row.sclk}")).decode("utf-8")
            f.write(build_request_line(few_shot_parts, pool["system_instruction"], row.sclk, image_b64, k,
                                       pool.get("temperature")) + "\n")
    print(f"Wrote {len(df)} requests ({os.path.getsize(path) / 1e6:.0f} MB) to {path}")
    return path


# ---------------------------------------------------------------------------
# Vertex AI batch prediction (same REST calls as the 3-P notebooks)
# ---------------------------------------------------------------------------
class ProgressFileReader:
    def __init__(self, file_obj, pbar):
        self.file_obj = file_obj
        self.pbar = pbar

    def read(self, size=-1):
        data = self.file_obj.read(size)
        self.pbar.update(len(data))
        return data

    def seek(self, offset, whence=0):
        return self.file_obj.seek(offset, whence)

    def tell(self):
        return self.file_obj.tell()


def _access_token():
    return subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def upload_jsonl(project_id, bucket_name, local_path, gcs_name):
    from google.cloud import storage
    bucket = storage.Client(project=project_id).bucket(bucket_name)
    blob = bucket.blob(gcs_name)
    blob.chunk_size = 10 * 1024 * 1024  # 10MB chunks for resumable upload
    with open(local_path, "rb") as f:
        with tqdm(total=os.path.getsize(local_path), unit="B", unit_scale=True, desc=f"Uploading {os.path.basename(local_path)}") as pbar:
            blob.upload_from_file(ProgressFileReader(f, pbar), content_type="application/json", timeout=3600)
    return f"gs://{bucket.name}/{gcs_name}"


def submit_batch_job(project_id, model, gcs_source, gcs_output_prefix, display_name):
    body = {
        "displayName": display_name,
        "model": f"publishers/google/models/{model}",
        "inputConfig": {"instancesFormat": "jsonl", "gcsSource": {"uris": [gcs_source]}},
        "outputConfig": {"predictionsFormat": "jsonl", "gcsDestination": {"outputUriPrefix": gcs_output_prefix}},
    }
    result = subprocess.run(
        ["curl", "-s", "-X", "POST",
         f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/batchPredictionJobs",
         "-H", f"Authorization: Bearer {_access_token()}",
         "-H", "Content-Type: application/json; charset=utf-8",
         "-d", "@-"],
        input=json.dumps(body), capture_output=True, text=True,
    )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        response = {}
    if result.returncode != 0 or "name" not in response:
        raise RuntimeError(f"Failed to submit batch job.\nstderr: {result.stderr}\nresponse: {result.stdout}")
    return response["name"]


# Terminal job states. A partially succeeded job still has (most of) its predictions.
FINISHED_STATES = ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED")
FAILED_STATES = ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED", "JOB_STATE_EXPIRED")


def get_batch_job(job_name):
    """Current status of a batch job (one GET request; {} if the response cannot be read)."""
    result = subprocess.run(
        ["curl", "-s", "-X", "GET", f"https://aiplatform.googleapis.com/v1/{job_name}",
         "-H", f"Authorization: Bearer {_access_token()}"],
        capture_output=True, text=True,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def wait_for_batch_job(job_name, poll_seconds=30):
    while True:
        status = get_batch_job(job_name)
        state = status.get("state", "UNKNOWN")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Job State: {state}")
        if state in FINISHED_STATES:
            return status
        if state in FAILED_STATES:
            raise RuntimeError(f"Job ended with state {state}. Details: {status.get('error', 'No error details.')}")
        time.sleep(poll_seconds)


def run_batch_job(project_id, bucket_name, model, local_jsonl_path, display_prefix):
    """Upload the request file, submit one Vertex AI batch job and wait until it finishes."""
    display_name = f"{display_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    bucket_name = bucket_name.replace("gs://", "")
    gcs_source = upload_jsonl(project_id, bucket_name, local_jsonl_path, f"input/3oh/{display_name}.jsonl")
    job_name = submit_batch_job(project_id, model, gcs_source, f"gs://{bucket_name}/output/3oh", display_name)
    print(f"Job Submitted! Name: {job_name}")
    return wait_for_batch_job(job_name)


_SCLK_PATTERN = re.compile(r"Sample ID \(sclk\): (\d+)")


def normalise_label(raw):
    if not isinstance(raw, str):
        return None
    label = raw.strip().strip("'\"")
    if label.lower().startswith("class "):
        label = label[6:].strip()
    key = label.lower().replace(" ", "").replace("_", "-")
    if key in ("3-oh", "3oh"):
        return LABEL_3OH
    if key in ("non-3-oh", "non-3oh", "non3-oh", "non3oh", "not-3-oh"):
        return LABEL_NON_3OH
    return label


def parse_prediction_line(line):
    """One output line of a batch job -> {sclk, predicted_label, explanation, error}."""
    obj = json.loads(line)
    sclk = None
    request_parts = obj.get("request", {}).get("contents", [{}])[-1].get("parts", [])
    for part in reversed(request_parts):
        match = _SCLK_PATTERN.search(part.get("text", ""))
        if match:
            sclk = int(match.group(1))
            break
    try:
        response_parts = obj["response"]["candidates"][0]["content"]["parts"]
        text = next(p["text"] for p in reversed(response_parts) if "text" in p and not p.get("thought"))
        answer = json.loads(text)
    except (KeyError, IndexError, TypeError, StopIteration, json.JSONDecodeError):
        return {"sclk": sclk, "predicted_label": None, "explanation": "",
                "error": obj.get("status") or "no parseable response"}
    if sclk is None and str(answer.get("id", "")).isdigit():
        sclk = int(answer["id"])
    return {"sclk": sclk, "predicted_label": normalise_label(answer.get("class")),
            "explanation": answer.get("explanation", ""), "error": None}


def iter_batch_predictions(project_id, bucket_name, job_status):
    """Stream and parse all prediction files of a finished batch job."""
    from google.cloud import storage
    gcs_output_dir = job_status.get("outputInfo", {}).get("gcsOutputDirectory")
    if not gcs_output_dir:
        raise ValueError("No gcsOutputDirectory found in job status data.")
    bucket = storage.Client(project=project_id).bucket(bucket_name.replace("gs://", ""))
    prefix = gcs_output_dir.replace(f"gs://{bucket.name}/", "").replace(f"gs://{bucket.name}", "")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    jsonl_blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".jsonl")]
    if not jsonl_blobs:
        raise FileNotFoundError(f"No prediction JSONL files found in GCS prefix: {prefix}")
    for blob in jsonl_blobs:
        with blob.open("rt") as f:
            with tqdm(total=blob.size, unit="B", unit_scale=True, desc=f"Downloading {os.path.basename(blob.name)}") as pbar:
                for line in f:
                    pbar.update(len(line.encode("utf-8")))
                    if line.strip():
                        yield parse_prediction_line(line)
