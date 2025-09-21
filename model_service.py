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

# (수정) 7-input 파이프라인 모델을 로드하도록 경로 변경
# Scaler가 파이프라인에 포함되어 있으므로 별도로 로드할 필요가 없습니다.
MODEL_PATHS = {
    "pipeline": os.path.join(model_dir, "mlp_pipeline_6input_magabs.pkl"), # Scikit-learn 파이프라인
    "label_encoder": os.path.join(model_dir, "label_encoder_6input.pkl"), # 라벨 인코더는 그대로 사용
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
    # (수정) 특징 컬럼에 'Mag_abs' 추가
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
        "model_name": "MLP 7-input Pipeline (with Full Calibration)", # (수정) 모델 이름 변경
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
    (수정) 7-input 파이프라인 모델에 맞게 예측을 수행합니다.
    """
    if not MODEL_LOADED:
        raise RuntimeError("Model components are not loaded properly.")

    # 1. 입력 데이터를 Pandas DataFrame으로 변환 (초기 6개 컬럼)
    # model_dump()는 Pydantic 모델을 dict로 변환합니다.
    initial_feature_cols = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
    feature_df = pd.DataFrame([data.model_dump()], columns=initial_feature_cols)

    # 2. Hard-Iron 보정 (기존과 동일)
    for col in ['Mag_X', 'Mag_Y', 'Mag_Z']:
        feature_df[col] -= models["hard_iron"][col]

    # 3. Soft-Iron 보정 (기존과 동일)
    mag_data = feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']].values
    feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']] = np.dot(mag_data, models["soft_iron"].T)
    
    # 4. Zero-Centering (기존과 동일)
    for col in initial_feature_cols:
        feature_df[col] -= models["zero_center"][col]
    
    # (추가) 5. 자기장 크기(Mag_abs) 계산 및 추가
    # 보정이 완료된 자기장 값으로 크기를 계산해야 합니다.
    feature_df['Mag_abs'] = np.sqrt(feature_df['Mag_X']**2 + feature_df['Mag_Y']**2 + feature_df['Mag_Z']**2)
    
    # (수정) 6. 파이프라인을 사용한 예측 (스케일링과 MLP 예측이 한 번에 처리됨)
    pipeline = models["pipeline"]
    
    # 학습 시 사용된 7개 컬럼 순서대로 DataFrame을 다시 정렬하여 예측
    final_features_df = feature_df[FEATURE_COLS]
    
    prediction_index = pipeline.predict(final_features_df)[0]
    
    # 최종 예측 결과에서 접미사 제거
    raw_prediction = models["label_encoder"].inverse_transform([prediction_index])[0]
    final_prediction = extract_location_id(raw_prediction)

    # 7. 신뢰도 및 Top-3 계산 (중복 제거 로직은 기존과 동일)
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
                unique_results.append((location_id, float(prob))) # prob를 float으로 변환
                
                if len(unique_results) >= 3:
                    break
        
        while len(unique_results) < 3:
            unique_results.append((0, 0.0))
        
        top_results_for_logging = unique_results[:3]
        top_results_for_response = unique_results[:3]
        
    return final_prediction, top_results_for_logging, top_results_for_response