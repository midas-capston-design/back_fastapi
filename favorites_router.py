# -*- coding: utf-8 -*-
# 이 파일은 머신러닝 모델 로딩 및 예측 로직을 전담합니다.

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
import pywt  # ✅ 웨이브렛 사용

# app.py가 아닌 여기서 직접 schemas를 import합니다.
import schemas

# --- 모델/전처리기 파일을 모두 로드 ---
base_dir = os.path.dirname(__file__)
model_dir = os.path.join(base_dir, "model")

# ✅ zero_center 제거(로드/사용 X)
MODEL_PATHS = {
    "mlp": os.path.join(model_dir, "mlp_model_6input.pkl"),
    "scaler": os.path.join(model_dir, "scaler.pkl"),  # Scaler를 별도로 로드
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
    # (유지) 특징 컬럼: 6-input
    FEATURE_COLS = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
    print("🚀 모든 모델 컴포넌트(6-input)가 성공적으로 로드되었습니다.")
except (FileNotFoundError, KeyError) as e:
    print(f"❌ 모델 로드 실패: {e}")

# --- 외부(app.py)에서 사용할 함수들 ---

def get_model_status() -> dict:
    """모델의 로드 상태와 정보를 반환합니다."""
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_name": "MLP 6-input (Hard/Soft-Iron + Wavelet + Scaler)",  # 이름 갱신
        "feature_cols": FEATURE_COLS if MODEL_LOADED else None,
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
    6-input 모델에 맞게 수동 전처리 및 예측을 수행합니다.
    순서: Hard-Iron → Soft-Iron → Wavelet(요청과 100% 동일 구현) → Scaler → MLP
    """
    if not MODEL_LOADED:
        raise RuntimeError("Model components are not loaded properly.")

    # 1. 입력 데이터를 Pandas DataFrame으로 변환
    feature_df = pd.DataFrame([data.model_dump()], columns=FEATURE_COLS)

    # (삭제) 자기장 크기(Mag_abs) 계산 로직 제거

    # 3. Hard-Iron 보정
    for col in ['Mag_X', 'Mag_Y', 'Mag_Z']:
        feature_df[col] = feature_df[col].astype(float) - float(models["hard_iron"][col])

    # 4. Soft-Iron 보정
    mag_data = feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']].values
    feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']] = np.dot(mag_data, models["soft_iron"].T)

    # ================================
    # ✅ 웨이브렛 Denoising (요청문과 100% 동일)
    # ================================
    # 호출부까지 동일하게 맞추기 위해 alias와 리스트명을 동일하게 만듭니다.
    df = feature_df
    mag_cols = ['Mag_X', 'Mag_Y', 'Mag_Z']

    # -------------------------------
    # 4️⃣ 웨이브렛 Denoising (Magnetometer)
    #    db4, level=3, soft-threshold(Donoho universal)
    # -------------------------------
    def wavelet_denoise(signal, wavelet='db4', level=3, mode='soft'):
        # DWT
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        # 노이즈 표준편차 추정 (최상위 detail 계수의 median absolute deviation)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) > 0 else 0.0
        if sigma == 0.0:
            return signal  # 노이즈 추정 불가 시 원신호 반환
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        # Approximation(A) 제외, Detail(D) 계수만 임계값 적용
        denoised_coeffs = [coeffs[0]] + [
            pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]
        ]
        recon = pywt.waverec(denoised_coeffs, wavelet=wavelet)
        # 길이 보정 (경계 처리로 길이가 1~2 샘플 달라질 수 있음)
        return recon[:len(signal)]

    for col in mag_cols:
        df[col] = wavelet_denoise(df[col].values, wavelet='db4', level=3, mode='soft')
    print("✅ Wavelet Denoising 완료")
    # ================================

    # (완전 제거) Zero-Centering 단계
    # for col in FEATURE_COLS:
    #     feature_df[col] -= models["zero_center"][col]

    # 6. 데이터 스케일링
    scaled_features = models["scaler"].transform(feature_df)

    # 7. MLP 모델로 예측 수행
    mlp_model = models["mlp"]
    prediction_index = mlp_model.predict(scaled_features)[0]

    # 최종 예측 결과에서 접미사 제거
    raw_prediction = models["label_encoder"].inverse_transform([prediction_index])[0]
    final_prediction = extract_location_id(raw_prediction)

    # 8. 신뢰도 및 Top-3 계산
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
