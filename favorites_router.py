# -*- coding: utf-8 -*-
# 이 파일은 머신러닝 모델 로딩 및 예측 로직을 전담합니다.

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
from collections import deque
import pywt  # 🔧 웨이브렛

# app.py가 아닌 여기서 직접 schemas를 import합니다.
import schemas

# --- 모델/전처리기 파일을 모두 로드 ---
base_dir = os.path.dirname(__file__)
model_dir = os.path.join(base_dir, "model")

# 🔧 zero_center 제거 (더 이상 사용/로드하지 않음)
MODEL_PATHS = {
    "mlp": os.path.join(model_dir, "mlp_model_6input.pkl"),
    "scaler": os.path.join(model_dir, "scaler.pkl"),
    "label_encoder": os.path.join(model_dir, "label_encoder_6input.pkl"),
    "soft_iron": os.path.join(model_dir, "soft_iron_matrix.pkl"),
    "hard_iron": os.path.join(model_dir, "bias.pkl"),
}

models = {}
MODEL_LOADED = False
FEATURE_COLS = None

try:
    for name, path in MODEL_PATHS.items():
        models[name] = joblib.load(path)
        print(f"✅ 모델 컴포넌트 로드 완료: {os.path.basename(path)}")

    MODEL_LOADED = True
    FEATURE_COLS = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
    print("🚀 모든 모델 컴포넌트(6-input)가 성공적으로 로드되었습니다.")
except (FileNotFoundError, KeyError) as e:
    print(f"❌ 모델 로드 실패: {e}")

# 🔧 웨이브렛 설정 & 버퍼
WAVELET_NAME = 'db4'
WAVELET_LEVEL = 3
WAVELET_MODE = 'soft'
BUFFER_SIZE = 64  # df 단위가 아니라 스트림 단위에서 최근 샘플을 모아 적용

# 자기장 3축에 대한 순환 버퍼 (실시간 단샘플 입력 대응)
_mag_buffers = {
    'Mag_X': deque(maxlen=BUFFER_SIZE),
    'Mag_Y': deque(maxlen=BUFFER_SIZE),
    'Mag_Z': deque(maxlen=BUFFER_SIZE),
}

def _wavelet_denoise_1d(arr: np.ndarray,
                        wavelet: str = WAVELET_NAME,
                        level: int = WAVELET_LEVEL,
                        mode: str = WAVELET_MODE) -> np.ndarray:
    """
    Donoho universal threshold 기반 1D 웨이브렛 디노이즈.
    입력: 1D ndarray
    출력: 1D ndarray (동일 길이)
    """
    if arr.size < 8:  # 길이가 너무 짧으면 웨이브렛 의미가 적으므로 원신호 반환
        return arr

    coeffs = pywt.wavedec(arr, wavelet=wavelet, level=level)
    # 노이즈 표준편차 추정 (최상위 detail 계수의 MAD)
    if len(coeffs[-1]) == 0:
        return arr
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if sigma == 0.0:
        return arr

    uthresh = sigma * np.sqrt(2 * np.log(len(arr)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]
    ]
    recon = pywt.waverec(denoised_coeffs, wavelet=wavelet)
    return recon[:len(arr)]

# --- 외부(app.py)에서 사용할 함수들 ---

def get_model_status() -> dict:
    """모델의 로드 상태와 정보를 반환합니다."""
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_name": "MLP 6-input (Hard/Soft-Iron + Wavelet + Scaler)",  # 🔧 설명 업데이트
        "feature_cols": FEATURE_COLS if MODEL_LOADED else None,
        "preprocess": {
            "hard_iron": True,
            "soft_iron": True,
            "wavelet": {"name": WAVELET_NAME, "level": WAVELET_LEVEL, "mode": WAVELET_MODE},
            "zero_centering": False,  # 🔧 제거
            "scaler": True,
        }
    }

def extract_location_id(label_str):
    """라벨에서 위치 ID만 추출하는 함수 (기존과 동일)"""
    try:
        return int(str(label_str).split('_')[0])
    except (ValueError, IndexError):
        try:
            return int(label_str)
        except:
            return 0

def run_prediction(data: schemas.SensorInput) -> Tuple[int, list, list]:
    """
    6-input 모델에 맞게 수동 전처리 및 예측 수행.
    전처리 순서: Hard-Iron → Soft-Iron → 🔧 Wavelet → Scaler → MLP
    """
    if not MODEL_LOADED:
        raise RuntimeError("Model components are not loaded properly.")

    # 1) 입력 → DataFrame
    feature_df = pd.DataFrame([data.model_dump()], columns=FEATURE_COLS)

    # 2) Hard-Iron 보정
    for col in ['Mag_X', 'Mag_Y', 'Mag_Z']:
        feature_df[col] = feature_df[col].astype(float) - float(models["hard_iron"][col])

    # 3) Soft-Iron 보정
    mag_data = feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']].values  # shape (1,3)
    feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']] = np.dot(mag_data, models["soft_iron"].T)

    # 4) 🔧 Wavelet Denoising (소프트아이언 뒤, 스케일러 전에)
    #    실시간 단샘플 예측 호환을 위해, 축별 버퍼에 누적 → 버퍼 전체를 디노이즈 → 마지막 값을 사용
    for col in ['Mag_X', 'Mag_Y', 'Mag_Z']:
        _mag_buffers[col].append(float(feature_df[col].iloc[0]))
        buf_arr = np.asarray(_mag_buffers[col], dtype=float)
        denoised = _wavelet_denoise_1d(buf_arr)
        denoised_last = denoised[-1] if denoised.size > 0 else float(feature_df[col].iloc[0])
        feature_df.at[0, col] = denoised_last

    # 🔧 (삭제) Zero-Centering 단계 완전 제거

    # 5) Scaler
    scaled_features = models["scaler"].transform(feature_df)

    # 6) MLP 예측
    mlp_model = models["mlp"]
    prediction_index = mlp_model.predict(scaled_features)[0]

    # 라벨 복원 및 위치 ID 추출
    raw_prediction = models["label_encoder"].inverse_transform([prediction_index])[0]
    final_prediction = extract_location_id(raw_prediction)

    # 7) 신뢰도 / Top-3 유니크 후보
    top_results_for_logging, top_results_for_response = [], []
    if hasattr(mlp_model, "predict_proba"):
        probabilities = mlp_model.predict_proba(scaled_features)[0]
        num_candidates = min(len(probabilities), 15)
        top_indices = np.argsort(-probabilities)[:num_candidates]
        raw_top_labels = models["label_encoder"].inverse_transform(top_indices)
        top_probs = probabilities[top_indices]

        seen_locations = set()
        unique_results = []
        for raw_label, prob in zip(raw_top_labels, top_probs):
            location_id = extract_location_id(raw_label)
            if location_id not in seen_locations:
                seen_locations.add(location_id)
                unique_results.append((location_id, float(prob)))
                if len(unique_results) >= 3:
                    break

        while len(unique_results) < 3:
            unique_results.append((0, 0.0))

        top_results_for_logging = unique_results[:3]
        top_results_for_response = unique_results[:3]

    return final_prediction, top_results_for_logging, top_results_for_response
