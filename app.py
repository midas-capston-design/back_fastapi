# -*- coding: utf-8 -*-
"""
FastAPI 서버: Scikit-learn Pipeline 위치 예측 + 사용자 관리 + 예측 위치 정보 (v5.0)
- 새로운 6-Input MLP 모델 및 다단계 보정 파이프라인 적용
"""
import os
import joblib
import numpy as np
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List

# --- 라우터, DB, 스키마, CRUD 함수 import ---
from database import engine, Base, get_db
from user_router import router as user_router
from favorites_router import router as favorites_router
from locations_router import router as locations_router
import crud
import schemas

# 서버 시작 시 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# 🔽 [수정] 새로운 모델 5개 로드 ---
# ------------------------------------
base_dir = os.path.dirname(__file__)
model_dir = os.path.join(base_dir, "model")

# 모델 파일 경로 정의
MODEL_PATHS = {
    "mlp": os.path.join(model_dir, "mlp_model_6input.pkl"),
    "scaler": os.path.join(model_dir, "scaler.pkl"),
    "zero_center": os.path.join(model_dir, "zero_center_means.pkl"),
    "soft_iron": os.path.join(model_dir, "soft_iron_matrix.pkl"),
    "hard_iron": os.path.join(model_dir, "bias.pkl")
}

# 모델 컴포넌트들을 담을 딕셔너리
models = {}
try:
    for name, path in MODEL_PATHS.items():
        models[name] = joblib.load(path)
        print(f"✅ 모델 컴포넌트 로드 완료: {os.path.basename(path)}")
    MODEL_LOADED = True
    # 학습 코드에 명시된 6개의 피처
    FEATURE_COLS = ['Mag_X', 'Mag_Y', 'Mag_Z', 'Ori_X', 'Ori_Y', 'Ori_Z']
    print("🚀 모든 모델 컴포넌트가 성공적으로 로드되었습니다.")
except (FileNotFoundError, KeyError) as e:
    print(f"❌ 모델 로드 실패: {e}")
    MODEL_LOADED = False
# ------------------------------------

# 🔽 [수정] 새로운 예측 파이프라인 Helper 함수 ---
def _run_new_prediction_pipeline(payload: schemas.SensorInput, top_k: int = 1):
    """새로운 모델의 다단계 보정 및 예측 파이프라인을 수행합니다."""
    
    # 1. 입력 데이터를 NumPy 배열로 변환
    mag_data = np.array([[payload.Mag_X, payload.Mag_Y, payload.Mag_Z]])
    ori_data = np.array([[payload.Ori_X, payload.Ori_Y, payload.Ori_Z]])

    # 2. 자기장 센서 데이터 보정 (Calibration)
    #    - Zero-centering -> Soft-iron correction -> Hard-iron correction
    mag_centered = mag_data - models["zero_center"]
    mag_soft_corrected = np.dot(mag_centered, models["soft_iron"])
    mag_calibrated = mag_soft_corrected - models["hard_iron"]

    # 3. 최종 피처 벡터 생성 (보정된 Mag 3축 + Ori 3축)
    feature_vector = np.concatenate((mag_calibrated, ori_data), axis=1)

    # 4. 데이터 스케일링 (StandardScaler)
    scaled_features = models["scaler"].transform(feature_vector)

    # 5. MLP 모델로 예측 수행
    mlp_model = models["mlp"]
    prediction = int(mlp_model.predict(scaled_features)[0])
    
    # 6. 신뢰도(Confidence) 및 Top-K 결과 계산
    confidence, top_k_list = None, None
    if hasattr(mlp_model, "predict_proba"):
        probabilities = mlp_model.predict_proba(scaled_features)
        confidence = float(np.max(probabilities))
        if top_k > 1:
            idx_sorted = np.argsort(-probabilities, axis=1)[:, :top_k]
            # label_encoder가 없으므로 정수 인덱스를 그대로 사용합니다.
            top_k_list = [f"{int(cls)} ({probabilities[0, cls]:.4f})" for cls in idx_sorted[0]]

    return prediction, confidence, top_k_list
# ----------------------------------------------------

# --- FastAPI 앱 초기화 및 라우터 등록 ---
app = FastAPI(
    title="Midas API - 실내 위치 예측 및 사용자 서비스 (6-Input MLP)",
    description="새로운 6-Input MLP 모델과 자기장 보정 파이프라인이 적용된 API 서버입니다.",
    version="5.0.0",
)
app.include_router(user_router, prefix="/users", tags=["users"])
app.include_router(favorites_router, prefix="/favorites", tags=["favorites"])
app.include_router(locations_router, prefix="/locations", tags=["Predicted Locations"])

# --- API Endpoints ---
@app.get("/health", summary="서버 상태 확인")
def health():
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_name": "MLPClassifier (6-Input with Calibration)",
        "feature_cols": FEATURE_COLS if MODEL_LOADED else None,
    }

@app.post("/predict", response_model=schemas.ModelOutput, summary="센서 데이터로 위치 예측")
def predict(data: schemas.SensorInput, db: Session = Depends(get_db)):
    if not MODEL_LOADED:
        return schemas.ModelOutput(
            prediction=-1, confidence=0.0, location_details=None,
            top_k_results=["Error: Model components are not loaded."]
        )
    
    # 1. [수정] 새로운 예측 파이프라인 실행
    prediction_result, confidence_score, top_k_list = _run_new_prediction_pipeline(data, top_k=data.top_k)
    
    # 2. DB에서 위치 정보 조회 (기존과 동일)
    location_info_db = crud.get_predicted_location(db, location_id=prediction_result)
    
    location_details_schema = None
    if location_info_db:
        location_details_schema = schemas.PredictedLocation.model_validate(location_info_db)

    # 3. 최종 응답 생성 (기존과 동일)
    return schemas.ModelOutput(
        prediction=prediction_result,
        confidence=confidence_score,
        location_details=location_details_schema,
        top_k_results=top_k_list
    )

@app.get("/", summary="API 정보")
def root():
    return {
        "message": "Full-featured Sensor Prediction API with User Management",
        "version": "5.0.0",
        "model_type": "6-Input MLPClassifier with Calibration Pipeline",
        # (기능 설명은 기존과 동일)
    }

