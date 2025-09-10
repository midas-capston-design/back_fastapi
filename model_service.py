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

def extract_location_id(label_str):
    """
    라벨에서 위치 ID만 추출하는 함수
    예: '25_0' -> 25, '1_2' -> 1, '7_3' -> 7
    """
    try:
        # '_' 기준으로 분리하고 첫 번째 부분만 사용
        return int(str(label_str).split('_')[0])
    except (ValueError, IndexError):
        # 분리할 수 없거나 숫자가 아닌 경우 원본을 숫자로 변환 시도
        try:
            return int(label_str)
        except:
            return 0

def run_prediction(data: schemas.SensorInput) -> Tuple[int, list, list]:
    """
    입력 데이터를 받아 보정 및 예측을 수행하고 결과를 반환합니다.
    접미사를 제거하고 중복을 제거하여 상위 3개 고유한 위치를 반환합니다.
    """
    # 1. 입력 데이터를 Pandas DataFrame으로 변환
    feature_df = pd.DataFrame([data.model_dump()], columns=FEATURE_COLS)

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
    
    # 최종 예측 결과에서 접미사 제거
    raw_prediction = models["label_encoder"].inverse_transform([prediction_index])[0]
    final_prediction = extract_location_id(raw_prediction)

    # 7. 신뢰도 및 Top-3 계산 (중복 제거)
    top_results_for_logging, top_results_for_response = [], []
    
    if hasattr(mlp_model, "predict_proba"):
        probabilities = mlp_model.predict_proba(scaled_features)[0]
        
        # 상위 10개 후보를 가져와서 중복 제거 후 3개 선택
        # (같은 위치 ID가 여러 접미사로 존재할 수 있으므로)
        num_candidates = min(len(probabilities), 15)  # 전체 클래스 수와 15 중 작은 값
        top_indices = np.argsort(-probabilities)[:num_candidates]
        
        raw_top_labels = models["label_encoder"].inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        # 중복 제거하면서 상위 3개 고유한 위치 선택
        seen_locations = set()
        unique_results = []
        
        for raw_label, prob in zip(raw_top_labels, top_probs):
            location_id = extract_location_id(raw_label)
            
            # 아직 보지 못한 위치 ID인 경우에만 추가
            if location_id not in seen_locations:
                seen_locations.add(location_id)
                unique_results.append((location_id, prob))
                
                # 3개를 모았으면 종료
                if len(unique_results) >= 3:
                    break
        
        # 3개 미만인 경우 빈 항목으로 채우기 (일반적으로 발생하지 않지만 안전장치)
        while len(unique_results) < 3:
            unique_results.append((0, 0.0))
        
        top_results_for_logging = unique_results[:3]
        top_results_for_response = unique_results[:3]
        
    return final_prediction, top_results_for_logging, top_results_for_response