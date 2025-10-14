# -*- coding: utf-8 -*-
# 온라인 전처리(윈도우+웨이브렛)와 100% 동일한 방식으로 실사용 예측

import os
import joblib
import numpy as np
from typing import Tuple, Optional, Dict, Any
import pywt
from collections import deque
import schemas

# -------------------------------
# 경로 및 파일 (수정 완료)
# -------------------------------
# 현재 파일의 위치를 기준으로 프로젝트 루트 폴더를 찾아서 모델 경로를 설정합니다.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(project_root, "model")

MODEL_PATHS = {
    "mlp": os.path.join(MODEL_DIR, "mlp_model_6input.pkl"),
    "scaler": os.path.join(MODEL_DIR, "scaler.pkl"),
    "label_encoder": os.path.join(MODEL_DIR, "label_encoder_6input.pkl"),
    "soft_iron": os.path.join(MODEL_DIR, "soft_iron_matrix.pkl"),
    "hard_iron": os.path.join(MODEL_DIR, "bias.pkl"),
    "preproc_params": os.path.join(MODEL_DIR, "preproc_params.pkl"),
    #"wavelet_sigma": os.path.join(MODEL_DIR, "wavelet_sigma.pkl"),
}

# -------------------------------
# 기본 파라미터
# -------------------------------
DEFAULT_PARAMS = {
    "wavelet": "db4",
    "level": 1,
    "thresh_mode": "soft",
    "window_size": 16,
    "hop_size": 1,
    "border_mode": "symmetric",
    "strategy": "adaptive",
    "warmup_policy": "pad",
    "min_fill_ratio": 0.5,
}

MAG_COLS = ["Mag_X", "Mag_Y", "Mag_Z"]
ORI_COLS = ["Ori_X", "Ori_Y", "Ori_Z"]
FEATURE_COLS = MAG_COLS + ORI_COLS

# -------------------------------
# 로드
# -------------------------------
models: Dict[str, Any] = {}
PREPROC_PARAMS = DEFAULT_PARAMS.copy()
FIXED_SIGMA = None
BUFFERS: Dict[str, deque] = {}
MODEL_LOADED = False

try:
    for k, p in MODEL_PATHS.items():
        if os.path.exists(p):
            models[k] = joblib.load(p)
            print(f"✅ 로드: {os.path.basename(p)}")

    required = ["mlp", "scaler", "label_encoder", "soft_iron", "hard_iron"]
    if not all(k in models for k in required):
        missing = [k for k in required if k not in models]
        raise FileNotFoundError(f"필수 모델 파일 누락: {missing}")

    if "preproc_params" in models:
        PREPROC_PARAMS.update(models["preproc_params"])
        print("✅ 전처리 파라미터 적용:", PREPROC_PARAMS)
    else:
        print("ℹ️ preproc_params.pkl 없음 → 기본 파라미터 사용:", PREPROC_PARAMS)

    if PREPROC_PARAMS.get("strategy", "adaptive") == "fixed" and "wavelet_sigma" in models:
        FIXED_SIGMA = models["wavelet_sigma"]
        print("✅ wavelet_sigma 적용(고정 임계):", {k: v.get("uthresh") for k, v in FIXED_SIGMA.items()})
    else:
        print("ℹ️ adaptive 모드 또는 wavelet_sigma 미존재")

    BUFFERS = {c: deque(maxlen=int(PREPROC_PARAMS["window_size"])) for c in MAG_COLS}
    MODEL_LOADED = True
    print("🚀 모든 모델 컴포넌트(6-input) 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")

# -------------------------------
# 유틸
# -------------------------------
def get_model_status() -> dict:
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_name": "MLP 6-input (Hard/Soft-Iron + Windowed Wavelet + Scaler)",
        "feature_cols": FEATURE_COLS if MODEL_LOADED else None,
        "wavelet_strategy": PREPROC_PARAMS.get("strategy"),
        "window_size": PREPROC_PARAMS.get("window_size"),
        "warmup_policy": PREPROC_PARAMS.get("warmup_policy"),
    }

def extract_location_id(label_str):
    try:
        return int(str(label_str).split("_")[0])
    except (ValueError, IndexError):
        try:
            return int(label_str)
        except Exception:
            return 0

def _wavelet_denoise_window(x: np.ndarray, col_name: str) -> np.ndarray:
    wavelet = PREPROC_PARAMS["wavelet"]
    level = int(PREPROC_PARAMS["level"])
    mode = PREPROC_PARAMS["thresh_mode"]
    border = PREPROC_PARAMS["border_mode"]
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=border)

    if PREPROC_PARAMS.get("strategy", "adaptive") == "fixed" and FIXED_SIGMA and FIXED_SIGMA.get(col_name):
        uthresh = float(FIXED_SIGMA[col_name].get("uthresh", 0.0))
        if uthresh <= 0: return x
    else:
        d = coeffs[-1]
        if len(d) == 0: return x
        sigma = np.median(np.abs(d)) / 0.6745
        if sigma == 0: return x
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))

    den = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]]
    y = pywt.waverec(den, wavelet=wavelet, mode=border)
    return y[:len(x)]

def _latest_value_with_warmup(col_name: str) -> float:
    window_size = int(PREPROC_PARAMS["window_size"])
    policy = PREPROC_PARAMS.get("warmup_policy", "pad")
    ratio = float(PREPROC_PARAMS.get("min_fill_ratio", 0.5))
    buf = np.array(BUFFERS[col_name], dtype=float)
    n = len(buf)

    if (policy == "wait" and n < window_size) or \
       (policy == "passthrough" and n < window_size) or \
       (policy == "ratio" and n < max(1, int(window_size * ratio))):
        return float(buf[-1]) if n > 0 else 0.0

    if n < window_size:
        x = np.pad(buf, (0, window_size - n), mode="edge") if n > 0 else np.zeros(window_size)
    else:
        x = buf[-window_size:]

    y = _wavelet_denoise_window(x, col_name)
    return float(y[-1])

# -------------------------------
# 예측 진입점
# -------------------------------
def run_prediction(data: schemas.SensorInput) -> Tuple[Optional[int], list, list]:
    if not MODEL_LOADED:
        raise RuntimeError("Model components are not loaded properly.")

    sample = data.model_dump()

    for c in MAG_COLS:
        sample[c] -= float(models["hard_iron"][c])

    mag_vec = np.array([sample[c] for c in MAG_COLS], dtype=float)
    mag_vec = models["soft_iron"] @ mag_vec
    for i, c in enumerate(MAG_COLS):
        sample[c] = mag_vec[i]

    for c in MAG_COLS:
        BUFFERS[c].append(sample[c])

    window_size = int(PREPROC_PARAMS["window_size"])
    policy = PREPROC_PARAMS.get("warmup_policy", "pad")
    n_now = min(len(BUFFERS[c]) for c in MAG_COLS)

    if policy == "wait" and n_now < window_size:
        return None, [], []

    latest = {c: _latest_value_with_warmup(c) for c in MAG_COLS}
    
    feature_vec = np.array([latest[c] for c in MAG_COLS] + [sample[c] for c in ORI_COLS]).reshape(1, -1)
    
    scaled = models["scaler"].transform(feature_vec)
    
    mlp = models["mlp"]
    pred_idx = mlp.predict(scaled)[0]
    
    raw_pred = models["label_encoder"].inverse_transform([pred_idx])[0]
    final_pred = extract_location_id(raw_pred)

    top_results_for_logging, top_results_for_response = [], []
    if hasattr(mlp, "predict_proba"):
        proba = mlp.predict_proba(scaled)[0]
        num_candidates = min(len(proba), 15)
        top_indices = np.argsort(-proba)[:num_candidates]
        raw_top = models["label_encoder"].inverse_transform(top_indices)
        top_probs = proba[top_indices]

        seen = set()
        uniq = []
        for lbl, p in zip(raw_top, top_probs):
            loc = extract_location_id(lbl)
            if loc not in seen:
                seen.add(loc)
                uniq.append((loc, float(p)))
                if len(uniq) >= 3:
                    break
        while len(uniq) < 3:
            uniq.append((0, 0.0))
        
        top_results_for_logging = uniq[:3]
        top_results_for_response = uniq[:3]

    return final_pred, top_results_for_logging, top_results_for_response