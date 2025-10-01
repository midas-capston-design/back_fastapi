# -*- coding: utf-8 -*-

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
import schemas

# --- 모델 경로 설정 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(project_root, "model")

MODEL_PATHS = {
    "mlp": os.path.join(model_dir, "mlp_model_6input.pkl"),
    "scaler": os.path.join(model_dir, "scaler.pkl"),
    "label_encoder": os.path.join(model_dir, "label_encoder_6input.pkl"),
    "zero_center": os.path.join(model_dir, "zero_center_means.pkl"),
    "soft_iron": os.path.join(model_dir, "soft_iron_matrix.pkl"),
    "hard_iron": os.path.join(model_dir, "bias.pkl")
}

FEATURE_COLS = None
models = {}
MODEL_LOADED = False

try:
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        models[name] = joblib.load(path)
        print(f"✅ 모델 컴포넌트 로드 완료: {os.path.basename(path)}")
    MODEL_LOADED = True
    FEATURE_COLS = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
    print("🚀 모든 모델 컴포넌트(6-input)가 성공적으로 로드되었습니다.")
except (FileNotFoundError, KeyError) as e:
    print(f"❌ 모델 로드 실패: {e}")


def extract_location_id(label_str):
    try:
        return int(str(label_str).split("_")[0])
    except (ValueError, IndexError):
        try:
            return int(label_str)
        except Exception:
            return 0


def run_prediction(data: schemas.SensorInput) -> Tuple[int, list, list]:
    if not MODEL_LOADED:
        raise RuntimeError("Model components are not loaded properly.")
    
    feature_df = pd.DataFrame([data.model_dump()], columns=FEATURE_COLS)

    # Hard-Iron 보정
    for col in ['Mag_X', 'Mag_Y', 'Mag_Z']:
        feature_df[col] -= models["hard_iron"][col]

    # Soft-Iron 보정
    mag_data = feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']].values
    feature_df[['Mag_X', 'Mag_Y', 'Mag_Z']] = np.dot(mag_data, models["soft_iron"].T)
    
    # Zero-Center 보정
    for col in FEATURE_COLS:
        feature_df[col] -= models["zero_center"][col]
    
    # 스케일링
    scaled_features = models["scaler"].transform(feature_df)
    
    # 예측
    mlp_model = models["mlp"]
    prediction_index = mlp_model.predict(scaled_features)[0]
    
    raw_prediction = models["label_encoder"].inverse_transform([prediction_index])[0]
    final_prediction = extract_location_id(raw_prediction)

    # Top-3 후보
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
