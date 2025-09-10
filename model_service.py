# -*- coding: utf-8 -*-
# 이 파일은 머신러닝 모델 로딩 및 예측 로직을 전담합니다.

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple

# app.py가 아닌 여기서 직접 schemas를 import합니다.
import schemas

# --- 6개의 모델/전처리기 파일을 모두 로드 ---
base_dir = os.path.dirname(__file__)
model_dir = os.path.join(base_dir, "model")

MODEL_PATHS = {
    "mlp": os.path.join(model_dir, "mlp_model_6input.pkl"),
    "label_encoder": os.path.join(model_dir, "label_encoder_6input.pkl"),
    "scaler": os.path.join(model_dir, "scaler.pkl"),
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
    FEATURE_COLS = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
    print("🚀 모든 모델 컴포넌트(보정 포함)가 성공적으로 로드되었습니다.")
except (FileNotFoundError, KeyError) as e:
    print(f"❌ 모델 로드 실패: {e}")

# --- 외부(app.py)에서 사용할 함수들 ---

def get_model_status() -> dict:
    """모델의 로드 상태와 정보를 반환합니다."""
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_name": "MLPClassifier (with Full Calibration Pipeline)",
        "feature_cols": FEATURE_COLS if MODEL_LOADED else None,
    }

def run_prediction(data: schemas.SensorInput) -> Tuple[int, list, list]:
    """
    입력 데이터를 받아 보정 및 예측을 수행하고 결과를 반환합니다.
    (기존 app.py의 run_full_prediction_pipeline 함수와 동일)
    """
    # 1. 입력 데이터를 Pandas DataFrame으로 변환
    feature_df = pd.DataFrame([data.model_dump(exclude={'top_k'})], columns=FEATURE_COLS)

    # 2. Hard-Iron 보정
    for col in ['Mag_X', 'Mag_Y', 'Mag_Z']:
        feature_df[col] -= models["hard_iron"][col]

    # 3. Soft-Iron 보정
    mag_data = feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']].values
    feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']] = np.dot(mag_data, models["soft_iron"].T)
    
    # 4. Zero-Centering
    for col in FEATURE_COLS:
        feature_df[col] -= models["zero_center"][col]
    
    # 5. 스케일링
    scaled_features = models["scaler"].transform(feature_df)
    
    # 6. MLP 예측 및 Label Decoding
    mlp_model = models["mlp"]
    prediction_index = mlp_model.predict(scaled_features)[0]
    final_prediction = int(models["label_encoder"].inverse_transform([prediction_index])[0])

    # 7. 신뢰도 및 Top-K 계산
    top_results_for_logging, top_results_for_response = [], []
    if hasattr(mlp_model, "predict_proba"):
        probabilities = mlp_model.predict_proba(scaled_features)[0]
        
        num_results_needed = max(3, data.top_k)
        top_indices = np.argsort(-probabilities)[:num_results_needed]
        
        top_labels = models["label_encoder"].inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        all_top_results = [(int(label), prob) for label, prob in zip(top_labels, top_probs)]

        top_results_for_logging = all_top_results[:3]
        top_results_for_response = all_top_results[:data.top_k]
        
    return final_prediction, top_results_for_logging, top_results_for_response

