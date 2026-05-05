"""
NMR Fusion Browser App
Cross-platform Gradio web UI for 1H + 13C NMR fusion model inference.
"""

import argparse
import glob
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Keep TensorFlow quiet-ish
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# -----------------------------
# Global app state
# -----------------------------
STATE = {
    "model": None,
    "label_map": None,
    "inv_label_map": None,
    "vector_len": 4096,
    "best_rule": "hybrid_perclass_count",
    "per_class_thresholds": None,
    "global_threshold": 0.5,
    "export_root": None,
}

DEFAULT_PPM_WINDOWS = {
    "1H": (-0.5, 12.5),
    "13C": (-5.0, 220.0),
}

# Fixed model export expected next to this file.
# Put your trained Colab export ZIP here with this exact name.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ZIP = PROJECT_ROOT / "nmr_artifacts_fusion.zip"
MODEL_CACHE_DIR = PROJECT_ROOT / ".model_cache"

# -----------------------------
# Utility helpers
# -----------------------------

def _first_existing(root: str, patterns: List[str]) -> Optional[str]:
    hits: List[str] = []
    for pat in patterns:
        hits.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
    hits = [h for h in hits if os.path.isfile(h)]
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]


def _extract_zip_to_temp(zip_path: str) -> str:
    out_dir = tempfile.mkdtemp(prefix="nmr_export_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def _safe_json_load(path: Optional[str]) -> Optional[dict]:
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Model loading
# -----------------------------

def load_fixed_model():
    """Load the fixed fusion model export from ./nmr_artifacts_fusion.zip."""
    try:
        src = DEFAULT_MODEL_ZIP

        if not src.exists():
            STATE.update({
                "model": None,
                "label_map": None,
                "inv_label_map": None,
                "per_class_thresholds": None,
                "export_root": None,
            })
            msg = (
                "⚠️ Model ZIP not found.\n\n"
                f"Expected file: `{src}`\n\n"
                "Place your trained Colab export ZIP in the project root and name it exactly "
                "`nmr_artifacts_fusion.zip`, then click **Reload model** or restart the app."
            )
            return msg, pd.DataFrame(columns=["compound", "index"])

        # Recreate cache so updated model ZIP is always used.
        if MODEL_CACHE_DIR.exists():
            shutil.rmtree(MODEL_CACHE_DIR)
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(MODEL_CACHE_DIR)

        export_root = str(MODEL_CACHE_DIR)

        keras_path = _first_existing(export_root, ["*.keras"])
        if keras_path is None:
            return "❌ Could not find a `.keras` model inside `nmr_artifacts_fusion.zip`.", pd.DataFrame()

        label_map_path = _first_existing(export_root, ["label_map.json"])
        if label_map_path is None:
            return "❌ Could not find `label_map.json` inside `nmr_artifacts_fusion.zip`.", pd.DataFrame()

        model = tf.keras.models.load_model(keras_path, compile=False)
        label_map_raw = _safe_json_load(label_map_path)
        label_map = {str(k): int(v) for k, v in label_map_raw.items()}
        inv_label_map = {int(v): str(k) for k, v in label_map.items()}

        cfg_path = _first_existing(export_root, ["model_config.json", "config.json", "*config*.json"])
        cfg = _safe_json_load(cfg_path) or {}

        vector_len = int(
            cfg.get("vector_len", cfg.get("input_len", cfg.get("VECTOR_LEN", 0)))
            or (model.input_shape[0][1] if isinstance(model.input_shape, list) else 4096)
            or 4096
        )
        best_rule = str(cfg.get("best_rule", "hybrid_perclass_count"))

        thresholds_path = _first_existing(export_root, ["thresholds.json", "*threshold*.json"])
        thr = _safe_json_load(thresholds_path) or {}
        best_rule = str(thr.get("best_rule", best_rule))
        global_threshold = float(
            thr.get("global_threshold_exact", thr.get("global_threshold", thr.get("best_t_exact", 0.5)))
        )

        per_class_thresholds = np.ones(len(label_map), dtype=np.float32) * global_threshold
        pct = thr.get("per_class_thresholds", None)
        if isinstance(pct, dict):
            for key, val in pct.items():
                if key in label_map:
                    per_class_thresholds[label_map[key]] = float(val)
                else:
                    try:
                        idx = int(key)
                        if 0 <= idx < len(per_class_thresholds):
                            per_class_thresholds[idx] = float(val)
                    except Exception:
                        pass
        elif isinstance(pct, list) and len(pct) == len(label_map):
            per_class_thresholds = np.asarray(pct, dtype=np.float32)

        STATE.update({
            "model": model,
            "label_map": label_map,
            "inv_label_map": inv_label_map,
            "vector_len": vector_len,
            "best_rule": best_rule,
            "per_class_thresholds": per_class_thresholds,
            "global_threshold": global_threshold,
            "export_root": export_root,
        })

        info = (
            "✅ Fixed model loaded from `nmr_artifacts_fusion.zip`\n\n"
            f"Model file inside ZIP: `{os.path.basename(keras_path)}`\n\n"
            f"Classes: **{len(label_map)}**\n\n"
            f"Vector length: **{vector_len}**\n\n"
            f"Default rule: **{best_rule}**\n\n"
            f"Global threshold: **{global_threshold:.3f}**"
        )
        label_df = pd.DataFrame({"compound": list(label_map.keys()), "index": list(label_map.values())}).sort_values("index")
        return info, label_df

    except Exception as e:
        STATE.update({"model": None})
        return f"❌ Loading failed: {type(e).__name__}: {e}", pd.DataFrame(columns=["compound", "index"])


# -----------------------------
# File parsing and preprocessing
# -----------------------------

def read_uploaded_table_or_vector(file_obj) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], str]:
    """
    Returns (df, vector, filename). For npy, df=None and vector is filled.
    For CSV/TXT, df is filled.
    """
    if file_obj is None:
        raise ValueError("No input file uploaded.")
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    name = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path).squeeze().astype(np.float32)
        if arr.ndim != 1:
            raise ValueError(f"{name}: .npy must contain a 1D vector.")
        return None, arr, name

    # Try robust CSV/TXT parsing
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        try:
            df = pd.read_csv(path, comment="#")
        except Exception:
            df = pd.read_table(path, header=None, comment="#")
    return df, None, name


def numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
    return cols


def get_col(df: pd.DataFrame, candidates: List[str], fallback_idx: Optional[int] = None) -> Optional[pd.Series]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        cand_lower = cand.lower()
        for lc, orig in lower_map.items():
            if lc == cand_lower or cand_lower in lc:
                return df[orig]
    if fallback_idx is not None:
        nums = numeric_columns(df)
        if len(nums) > fallback_idx:
            return df[nums[fallback_idx]]
    return None


def pascal_coefficients(n: int) -> np.ndarray:
    n = int(max(0, n))
    row = [1]
    for _ in range(n):
        row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]
    arr = np.asarray(row, dtype=np.float32)
    return arr / arr.max()


def lorentzian(x: np.ndarray, center: float, hwhm: float) -> np.ndarray:
    hwhm = max(float(hwhm), 1e-6)
    return (hwhm * hwhm) / ((x - float(center)) ** 2 + hwhm * hwhm)


def normalize_spectrum(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).copy()
    y = y - np.median(y)
    # If peaks are negative, flip. This helps phase-inverted exports.
    if abs(float(np.min(y))) > abs(float(np.max(y))) * 1.2:
        y = -y
    p995 = np.percentile(np.abs(y), 99.5)
    if p995 > 0:
        y = np.clip(y, -p995, p995)
    denom = np.max(np.abs(y)) + 1e-8
    y = y / denom
    return y.astype(np.float32)


def resample_vector(y: np.ndarray, vector_len: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).squeeze()
    if y.ndim != 1 or len(y) < 2:
        raise ValueError("Vector must be 1D and contain at least 2 points.")
    if len(y) == vector_len:
        return normalize_spectrum(y)
    old_x = np.linspace(0.0, 1.0, len(y), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, vector_len, dtype=np.float32)
    return normalize_spectrum(np.interp(new_x, old_x, y).astype(np.float32))


def remove_artifacts(axis: np.ndarray, y: np.ndarray, nucleus: str, remove: bool) -> np.ndarray:
    if not remove:
        return y
    windows = []
    if nucleus == "1H":
        windows = [(-0.05, 0.05), (4.65, 4.90), (7.22, 7.30)]
    elif nucleus == "13C":
        windows = [(-0.5, 0.5), (76.5, 77.5)]
    y = y.copy()
    for lo, hi in windows:
        mask = (axis >= lo) & (axis <= hi)
        if np.any(mask):
            y[mask] = np.median(y)
    return y


def preprocess_xy(df: pd.DataFrame, nucleus: str, vector_len: int, remove_common_artifacts: bool) -> Tuple[np.ndarray, np.ndarray, str]:
    ppm_s = get_col(df, ["ppm", "shift", "chemical_shift", "x"], fallback_idx=0)
    int_s = get_col(df, ["intensity", "int", "y", "amplitude", "abs"], fallback_idx=1)
    if ppm_s is None or int_s is None:
        raise ValueError("XY mode needs two numeric columns: ppm/intensity or x/intensity.")
    ppm = pd.to_numeric(ppm_s, errors="coerce").to_numpy(np.float32)
    inten = pd.to_numeric(int_s, errors="coerce").to_numpy(np.float32)
    mask = np.isfinite(ppm) & np.isfinite(inten)
    ppm, inten = ppm[mask], inten[mask]
    ppm_min, ppm_max = DEFAULT_PPM_WINDOWS[nucleus]
    mask = (ppm >= ppm_min) & (ppm <= ppm_max)
    ppm, inten = ppm[mask], inten[mask]
    if len(ppm) < 2:
        raise ValueError(f"{nucleus}: too few XY points inside ppm window {ppm_min}..{ppm_max}.")
    order = np.argsort(ppm)
    ppm, inten = ppm[order], inten[order]
    axis = np.linspace(ppm_min, ppm_max, vector_len, dtype=np.float32)
    y = np.interp(axis, ppm, inten).astype(np.float32)
    y = remove_artifacts(axis, y, nucleus, remove_common_artifacts)
    return axis, normalize_spectrum(y), "xy"


def preprocess_peaklist(
    df: pd.DataFrame,
    nucleus: str,
    vector_len: int,
    h_mhz: float,
    c_mhz: float,
    remove_common_artifacts: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    ppm_s = get_col(df, ["ppm", "shift", "chemical_shift"], fallback_idx=0)
    int_s = get_col(df, ["intensity", "int", "relative", "area", "integral", "amplitude"], fallback_idx=1)
    if ppm_s is None:
        raise ValueError("Peak list mode needs at least a ppm column.")
    ppm = pd.to_numeric(ppm_s, errors="coerce").to_numpy(np.float32)
    if int_s is not None:
        inten = pd.to_numeric(int_s, errors="coerce").fillna(1.0).to_numpy(np.float32)
    else:
        inten = np.ones_like(ppm, dtype=np.float32)
    mask = np.isfinite(ppm) & np.isfinite(inten)
    ppm, inten = ppm[mask], inten[mask]

    ppm_min, ppm_max = DEFAULT_PPM_WINDOWS[nucleus]
    mask = (ppm >= ppm_min) & (ppm <= ppm_max)
    ppm, inten = ppm[mask], inten[mask]
    if len(ppm) == 0:
        raise ValueError(f"{nucleus}: no peaks inside ppm window {ppm_min}..{ppm_max}.")

    axis = np.linspace(ppm_min, ppm_max, vector_len, dtype=np.float32)
    y = np.zeros_like(axis, dtype=np.float32)

    # Optional peaklist columns
    n_s = get_col(df, ["n_neighbors", "neighbors", "n", "multiplicity_n"], fallback_idx=None)
    j_s = get_col(df, ["j_hz_typical", "j_hz", "j", "coupling"], fallback_idx=None)
    lw_s = get_col(df, ["linewidth", "hwhm", "width"], fallback_idx=None)
    ex_s = get_col(df, ["exchangeable", "broad"], fallback_idx=None)

    if n_s is not None:
        n_vals = pd.to_numeric(n_s, errors="coerce").fillna(0).to_numpy()
        n_vals = n_vals[mask] if len(n_vals) == len(mask) else np.zeros(len(ppm))
    else:
        n_vals = np.zeros(len(ppm))

    if j_s is not None:
        j_vals = pd.to_numeric(j_s, errors="coerce").fillna(7.0).to_numpy()
        j_vals = j_vals[mask] if len(j_vals) == len(mask) else np.ones(len(ppm)) * 7.0
    else:
        j_vals = np.ones(len(ppm)) * 7.0

    if lw_s is not None:
        lw_vals = pd.to_numeric(lw_s, errors="coerce").fillna(np.nan).to_numpy()
        lw_vals = lw_vals[mask] if len(lw_vals) == len(mask) else np.ones(len(ppm)) * np.nan
    else:
        lw_vals = np.ones(len(ppm)) * np.nan

    if ex_s is not None:
        ex_vals_all = ex_s.astype(str).to_numpy()
        ex_vals = ex_vals_all[mask] if len(ex_vals_all) == len(mask) else np.array(["False"] * len(ppm))
    else:
        ex_vals = np.array(["False"] * len(ppm))

    # intensity scale
    inten = np.asarray(inten, dtype=np.float32)
    if abs(float(np.min(inten))) > abs(float(np.max(inten))) * 1.2:
        inten = -inten
    if np.max(np.abs(inten)) > 0:
        inten = inten / (np.max(np.abs(inten)) + 1e-8)
    inten = np.clip(inten, 0.0, None)
    if np.max(inten) == 0:
        inten = np.ones_like(inten, dtype=np.float32)

    for center, amp, n_nei, j_hz, lw, ex in zip(ppm, inten, n_vals, j_vals, lw_vals, ex_vals):
        n_nei = int(max(0, round(float(n_nei)))) if nucleus == "1H" else 0
        coeffs = pascal_coefficients(n_nei)

        if nucleus == "1H":
            spectrometer_mhz = max(float(h_mhz), 1.0)
            split_ppm = float(j_hz) / spectrometer_mhz
            default_hwhm = 0.006
            if _boolish(ex):
                default_hwhm = 0.030
        else:
            split_ppm = 0.0
            default_hwhm = 0.28

        hwhm = float(lw) if np.isfinite(lw) and float(lw) > 0 else default_hwhm
        offsets = (np.arange(len(coeffs)) - (len(coeffs) - 1) / 2.0) * split_ppm
        for off, coef in zip(offsets, coeffs):
            y += float(amp) * float(coef) * lorentzian(axis, float(center) + float(off), hwhm)

    y = remove_artifacts(axis, y, nucleus, remove_common_artifacts)
    return axis, normalize_spectrum(y), "peaklist"


def preprocess_input_file(
    file_obj,
    nucleus: str,
    input_mode: str,
    vector_len: int,
    h_mhz: float,
    c_mhz: float,
    remove_common_artifacts: bool,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    df, vector, filename = read_uploaded_table_or_vector(file_obj)
    mode = str(input_mode).lower().strip()

    if vector is not None:
        axis = np.arange(vector_len, dtype=np.float32)
        return axis, resample_vector(vector, vector_len), "vector", filename

    assert df is not None
    nums = numeric_columns(df)
    if not nums:
        raise ValueError(f"{filename}: no numeric columns found.")

    if mode == "auto":
        cols_lower = [str(c).lower() for c in df.columns]
        has_ppm = any("ppm" in c or "shift" in c for c in cols_lower)
        has_mult = any("n_neighbor" in c or "j_hz" in c or "coupling" in c for c in cols_lower)
        if has_mult or (has_ppm and len(df) < 600):
            mode = "peak list"
        elif len(nums) == 1:
            mode = "vector"
        else:
            # dense data with two columns defaults to XY spectrum
            mode = "xy spectrum"

    if mode == "vector":
        y = pd.to_numeric(df[nums[0]], errors="coerce").dropna().to_numpy(np.float32)
        axis = np.arange(vector_len, dtype=np.float32)
        return axis, resample_vector(y, vector_len), "vector", filename

    if mode == "xy spectrum":
        axis, vec, kind = preprocess_xy(df, nucleus, vector_len, remove_common_artifacts)
        return axis, vec, kind, filename

    if mode == "peak list":
        axis, vec, kind = preprocess_peaklist(df, nucleus, vector_len, h_mhz, c_mhz, remove_common_artifacts)
        return axis, vec, kind, filename

    raise ValueError(f"Unknown input mode: {input_mode}")


# -----------------------------
# Plotting
# -----------------------------

def make_plot(axis_h: np.ndarray, xh: np.ndarray, kind_h: str, axis_c: np.ndarray, xc: np.ndarray, kind_c: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(axis_h, xh)
    axes[0].set_title(f"1H input ({kind_h})")
    axes[0].set_xlabel("ppm" if kind_h in {"xy", "peaklist"} else "index")
    axes[0].set_ylabel("normalized intensity")
    axes[0].grid(alpha=0.25)
    if kind_h in {"xy", "peaklist"}:
        axes[0].invert_xaxis()

    axes[1].plot(axis_c, xc)
    axes[1].set_title(f"13C input ({kind_c})")
    axes[1].set_xlabel("ppm" if kind_c in {"xy", "peaklist"} else "index")
    axes[1].set_ylabel("normalized intensity")
    axes[1].grid(alpha=0.25)
    if kind_c in {"xy", "peaklist"}:
        axes[1].invert_xaxis()

    plt.tight_layout()
    return fig


# -----------------------------
# Inference
# -----------------------------

def unpack_prediction(pred, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(pred, dict):
        vals = list(pred.values())
        if "labels" in pred:
            labels = np.asarray(pred["labels"])
        else:
            labels = np.asarray(vals[np.argmax([v.shape[-1] for v in vals])])
        if "count" in pred:
            count = np.asarray(pred["count"])
        else:
            count = np.asarray(vals[np.argmin([v.shape[-1] for v in vals])])
        return labels, count
    if isinstance(pred, (tuple, list)):
        a, b = np.asarray(pred[0]), np.asarray(pred[1])
        return (a, b) if a.shape[-1] == n_classes else (b, a)
    raise ValueError("Unknown model prediction output.")


def predict_labels(probs: np.ndarray, count_probs: np.ndarray, rule: str, manual_threshold: float) -> np.ndarray:
    n_classes = len(probs)
    thresholds = STATE["per_class_thresholds"]
    if thresholds is None or len(thresholds) != n_classes:
        thresholds = np.ones(n_classes, dtype=np.float32) * float(manual_threshold)
    k = int(np.argmax(count_probs) + 1)
    order = np.argsort(probs)[::-1]
    yhat = np.zeros(n_classes, dtype=np.int32)

    rule = rule.lower().strip()
    if rule == "auto":
        rule = str(STATE["best_rule"]).lower().strip()

    if rule == "global threshold":
        yhat = (probs >= float(manual_threshold)).astype(np.int32)
        if yhat.sum() == 0:
            yhat[order[0]] = 1
        return yhat

    if rule == "per-class threshold":
        yhat = (probs >= thresholds).astype(np.int32)
        if yhat.sum() == 0:
            yhat[order[0]] = 1
        return yhat

    if rule in {"calibrated_topk_by_count", "calibrated top-k by count"}:
        scores = probs - thresholds
        order2 = np.argsort(scores)[::-1]
        yhat[order2[:k]] = 1
        return yhat

    if rule in {"hybrid_perclass_count", "hybrid per-class + count"}:
        yhat = (probs >= thresholds).astype(np.int32)
        if int(yhat.sum()) != k:
            yhat[:] = 0
            yhat[order[:k]] = 1
        if yhat.sum() == 0:
            yhat[order[0]] = 1
        return yhat

    # default top-k by count
    yhat[order[:k]] = 1
    return yhat


def rank_expected(expected: List[str], probs: np.ndarray) -> str:
    inv_map = STATE["inv_label_map"]
    label_map = STATE["label_map"]
    order = np.argsort(probs)[::-1]
    rank_by_idx = {int(idx): rank + 1 for rank, idx in enumerate(order)}
    pieces = []
    for label in expected:
        label = label.strip()
        if not label:
            continue
        if label not in label_map:
            pieces.append(f"{label}:not_in_label_map")
            continue
        idx = label_map[label]
        pieces.append(f"{label}:rank={rank_by_idx[idx]},p={float(probs[idx]):.4f}")
    return "; ".join(pieces)


def run_prediction(
    h_file,
    c_file,
    input_mode,
    inference_rule,
    manual_threshold,
    expected_labels_text,
    h_mhz,
    c_mhz,
    remove_common_artifacts,
    try_flipped,
):
    if STATE["model"] is None:
        raise gr.Error("Load a model first.")
    if h_file is None or c_file is None:
        raise gr.Error("Upload both 1H and 13C files.")

    vector_len = int(STATE["vector_len"])
    axis_h, xh, kind_h, name_h = preprocess_input_file(h_file, "1H", input_mode, vector_len, h_mhz, c_mhz, remove_common_artifacts)
    axis_c, xc, kind_c, name_c = preprocess_input_file(c_file, "13C", input_mode, vector_len, h_mhz, c_mhz, remove_common_artifacts)
    fig = make_plot(axis_h, xh, kind_h, axis_c, xc, kind_c)

    def _one_predict(xh_use, xc_use, flipped_label: str):
        XH = xh_use.reshape(1, vector_len, 1).astype(np.float32)
        XC = xc_use.reshape(1, vector_len, 1).astype(np.float32)
        pred = STATE["model"].predict([XH, XC], verbose=0)
        label_probs, count_probs = unpack_prediction(pred, len(STATE["label_map"]))
        probs = label_probs[0].astype(np.float32)
        cprobs = count_probs[0].astype(np.float32)
        yhat = predict_labels(probs, cprobs, inference_rule, manual_threshold)
        return probs, cprobs, yhat, flipped_label

    candidates = [_one_predict(xh, xc, "normal")]
    if try_flipped:
        candidates.append(_one_predict(xh[::-1].copy(), xc[::-1].copy(), "flipped"))

    # If flipped is tried, choose candidate with larger mean detected probability.
    best = None
    best_score = -np.inf
    for probs, cprobs, yhat, tag in candidates:
        det = np.where(yhat == 1)[0]
        score = float(np.mean(probs[det])) if len(det) else float(np.max(probs))
        if score > best_score:
            best_score = score
            best = (probs, cprobs, yhat, tag)
    probs, cprobs, yhat, chosen_variant = best

    inv = STATE["inv_label_map"]
    detected_idx = np.where(yhat == 1)[0].tolist()
    detected_rows = []
    for i in detected_idx:
        detected_rows.append({
            "compound": inv[int(i)],
            "probability": float(probs[int(i)]),
        })
    detected_df = pd.DataFrame(detected_rows).sort_values("probability", ascending=False) if detected_rows else pd.DataFrame(columns=["compound", "probability"])

    order = np.argsort(probs)[::-1]
    top_rows = []
    for rank, i in enumerate(order[:20], start=1):
        top_rows.append({
            "rank": rank,
            "detected": bool(i in detected_idx),
            "compound": inv[int(i)],
            "probability": float(probs[int(i)]),
        })
    top_df = pd.DataFrame(top_rows)

    expected = [x.strip() for x in str(expected_labels_text or "").replace(";", ",").split(",") if x.strip()]
    detected_names = [inv[int(i)] for i in detected_idx]
    if expected:
        missed = sorted(set(expected) - set(detected_names))
        extra = sorted(set(detected_names) - set(expected))
        expected_df = pd.DataFrame([{
            "expected_labels": ", ".join(expected),
            "detected_labels": ", ".join(detected_names),
            "exact_match": int(set(expected) == set(detected_names)),
            "missed_labels": ", ".join(missed),
            "extra_labels": ", ".join(extra),
            "expected_rank": rank_expected(expected, probs),
        }])
    else:
        expected_df = pd.DataFrame(columns=["expected_labels", "detected_labels", "exact_match", "missed_labels", "extra_labels", "expected_rank"])

    count_lines = [f"{i+1} component(s): {float(p):.4f}" for i, p in enumerate(cprobs)]
    summary = (
        f"Input files: {name_h} + {name_c}\n"
        f"Preprocessing: 1H={kind_h}, 13C={kind_c}\n"
        f"Chosen axis variant: {chosen_variant}\n"
        f"Inference rule: {inference_rule} (model default: {STATE['best_rule']})\n"
        f"Predicted n_components: {int(np.argmax(cprobs)+1)}\n"
        f"Count probabilities: {' | '.join(count_lines)}\n\n"
        "Detected compounds:\n" +
        ("\n".join([f"- {r['compound']}: {r['probability']:.4f}" for r in detected_rows]) if detected_rows else "None")
    )

    # Save latest prepared vectors for user convenience
    latest_dir = tempfile.mkdtemp(prefix="nmr_vectors_")
    np.save(os.path.join(latest_dir, "prepared_1H_vector.npy"), xh.astype(np.float32))
    np.save(os.path.join(latest_dir, "prepared_13C_vector.npy"), xc.astype(np.float32))
    top_df.to_csv(os.path.join(latest_dir, "top_probabilities.csv"), index=False)
    detected_df.to_csv(os.path.join(latest_dir, "detected_compounds.csv"), index=False)
    expected_df.to_csv(os.path.join(latest_dir, "expected_check.csv"), index=False)
    archive_path = shutil.make_archive(latest_dir, "zip", latest_dir)

    return fig, summary, detected_df, top_df, expected_df, archive_path


# -----------------------------
# UI
# -----------------------------

def build_ui():
    initial_status, initial_labels = load_fixed_model()

    with gr.Blocks(title="NMR Spectra Analysis Model") as demo:
        gr.Markdown("# NMR Spectra Analysis Model")
        gr.Markdown(
            "Local browser showcase for the trained **¹H + ¹³C fusion NMR model**. "
            "The app automatically loads `nmr_artifacts_fusion.zip` from the project root. "
            "Upload spectra/vectors, plot prepared inputs, and show predicted compounds."
        )

        with gr.Accordion("Model status", open=True):
            model_status = gr.Markdown(initial_status)
            with gr.Row():
                reload_btn = gr.Button("Reload model", variant="secondary")
            label_table = gr.Dataframe(value=initial_labels, label="Loaded label map", interactive=False)
            reload_btn.click(load_fixed_model, inputs=[], outputs=[model_status, label_table])

        with gr.Tab("Predict"):
            with gr.Row():
                h_file = gr.File(label="Upload ¹H vector / XY / peak list")
                c_file = gr.File(label="Upload ¹³C vector / XY / peak list")

            with gr.Row():
                input_mode = gr.Dropdown(
                    choices=["Auto", "Vector", "XY spectrum", "Peak list"],
                    value="Auto",
                    label="Input mode",
                )
                inference_rule = gr.Dropdown(
                    choices=[
                        "Auto",
                        "hybrid_perclass_count",
                        "topk_by_count",
                        "calibrated_topk_by_count",
                        "global threshold",
                        "per-class threshold",
                    ],
                    value="Auto",
                    label="Inference rule",
                )
                manual_threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Manual/global threshold")

            with gr.Accordion("Advanced preprocessing", open=False):
                with gr.Row():
                    h_mhz = gr.Number(value=400.0, label="¹H spectrometer MHz for J splitting")
                    c_mhz = gr.Number(value=100.0, label="¹³C spectrometer MHz")
                with gr.Row():
                    remove_common_artifacts = gr.Checkbox(value=False, label="Remove common solvent/TMS artifact windows")
                    try_flipped = gr.Checkbox(value=False, label="Try flipped vector axis too")
                expected_labels = gr.Textbox(
                    label="Expected labels for sanity check, comma separated",
                    placeholder="ethanol, acetone",
                )

            predict_btn = gr.Button("Predict", variant="primary")

            with gr.Row():
                plot_out = gr.Plot(label="Prepared input vectors")
                summary_out = gr.Textbox(label="Prediction summary", lines=14)

            with gr.Row():
                detected_out = gr.Dataframe(label="Detected compounds", interactive=False)
                top_out = gr.Dataframe(label="Top probabilities", interactive=False)

            expected_out = gr.Dataframe(label="Expected-label check", interactive=False)
            download_out = gr.File(label="Download prepared vectors + CSV outputs")

            predict_btn.click(
                run_prediction,
                inputs=[
                    h_file,
                    c_file,
                    input_mode,
                    inference_rule,
                    manual_threshold,
                    expected_labels,
                    h_mhz,
                    c_mhz,
                    remove_common_artifacts,
                    try_flipped,
                ],
                outputs=[plot_out, summary_out, detected_out, top_out, expected_out, download_out],
            )

        with gr.Tab("Input format help"):
            gr.Markdown(
                """
## Required model file

Place your trained Colab export ZIP in the project root with this exact name:

```text
nmr_artifacts_fusion.zip
```

The app loads this file automatically. There is no model picker in the interface.

## Accepted input file types

### `.npy` vector
A 1D NumPy vector. If its length differs from the model input length, it is interpolated.

### Vector CSV/TXT
One numeric column with intensities.

```csv
intensity
0.0
0.02
0.15
```

### XY spectrum CSV/TXT
Two numeric columns. For NMR, the first column should be ppm and the second intensity.

```csv
ppm,intensity
7.26,0.8
7.25,1.0
```

### Peak list CSV/TXT
Peak centers and relative intensities. For ¹H, optional multiplicity fields help a lot.

```csv
ppm,intensity,n_neighbors,j_hz_typical,exchangeable
3.66,0.667,3,7.1,False
1.18,1.000,2,7.1,False
```

`n_neighbors=3` produces a quartet. `n_neighbors=2` produces a triplet.

## Why center-only real peak lists can fail

A center-only ¹H peak list gives only singlet-like peaks. The model was trained with generated multiplet patterns, so molecules such as ethanol are much easier when the ¹H peak list includes `n_neighbors` and `j_hz_typical`.
                """
            )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_ui()
    # show_api=False avoids a known Gradio 4.x / Pydantic schema parsing path
    # that can crash with: TypeError: argument of type 'bool' is not iterable.
    try:
        app.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
            show_api=False,
        )
    except TypeError:
        # Older Gradio fallback if show_api is not supported.
        app.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
        )
