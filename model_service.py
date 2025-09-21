# -*- coding: utf-8 -*-
# 이 파일은 머신러닝 모델 로딩 및 예측 로직을 전담합니다.

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple

# app.py가 아닌 여기서 직접 schemas를 import합니다.
import schemas

# --- 모델/전처리기 파일을 모두 로드 ---
base_dir = os.path.dirname(__file__)
model_dir = os.path.join(base_dir, "model")

MODEL_PATHS = {
    "pipeline": os.path.join(model_dir, "mlp_pipeline_6input_magabs.pkl"),
    "label_encoder": os.path.join(model_dir, "label_encoder_6input.pkl"),
    "zero_center": os.path.join(model_dir, "zero_center_means.pkl"),
    "soft_iron": os.path.join(model_dir, "soft_iron_matrix.pkl"),
    "hard_iron": os.path.join(model_dir, "bias.pkl")
}

models = {}
MODEL_LOADED = False
FEATURE_COLS = None
try:
    for name, path in MODEL_PATHS.items():
        models[name] = joblib.load(path)
        print(f"✅ 모델 컴포넌트 로드 완료: {os.path.basename(path)}")
    MODEL_LOADED = True
    FEATURE_COLS = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z', 'Mag_abs']
    print("🚀 모든 모델 컴포넌트(7-input Pipeline)가 성공적으로 로드되었습니다.")
except (FileNotFoundError, KeyError) as e:
    print(f"❌ 모델 로드 실패: {e}")

# --- 외부(app.py)에서 사용할 함수들 ---

def get_model_status() -> dict:
    """모델의 로드 상태와 정보를 반환합니다."""
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_name": "MLP 7-input Pipeline (with Full Calibration)",
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
    (수정) Mag_abs를 보정 전에 먼저 계산하도록 로직 순서 변경
    """
    if not MODEL_LOADED:
        raise RuntimeError("Model components are not loaded properly.")

    # 1. 입력 데이터를 Pandas DataFrame으로 변환
    initial_feature_cols = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
    feature_df = pd.DataFrame([data.model_dump()], columns=initial_feature_cols)

    # (수정 ✨) 2. 자기장 크기(Mag_abs)를 "보정 전" 원본 값으로 먼저 계산
    feature_df['Mag_abs'] = np.sqrt(feature_df['Mag_X']**2 + feature_df['Mag_Y']**2 + feature_df['Mag_Z']**2)

    # 3. Hard-Iron 보정 (이후 과정은 기존과 동일)
    for col in ['Mag_X', 'Mag_Y', 'Mag_Z']:
        feature_df[col] -= models["hard_iron"][col]

    # 4. Soft-Iron 보정
    mag_data = feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']].values
    feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']] = np.dot(mag_data, models["soft_iron"].T)
    
    # 5. Zero-Centering
    for col in initial_feature_cols:
        feature_df[col] -= models["zero_center"][col]
    
    # 6. 파이프라인을 사용한 예측
    pipeline = models["pipeline"]
    final_features_df = feature_df[FEATURE_COLS]
    prediction_index = pipeline.predict(final_features_df)[0]
    
    # 최종 예측 결과에서 접미사 제거
    raw_prediction = models["label_encoder"].inverse_transform([prediction_index])[0]
    final_prediction = extract_location_id(raw_prediction)

    # 7. 신뢰도 및 Top-3 계산
    top_results_for_logging, top_results_for_response = [], []
    
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(final_features_df)[0]
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