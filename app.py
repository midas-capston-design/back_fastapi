# -*- coding: utf-8 -*-
"""
FastAPI 서버: Scikit-learn Pipeline 위치 예측 + 사용자 관리 + 예측 위치 정보 (v9.0)
- 모델 로딩 및 예측 로직을 model_service.py로 분리하여 구조 개선
"""
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import logging

# --- 라우터, DB, 스키마, CRUD 함수 import ---
from database import engine, Base, get_db
from user_router import router as user_router
from favorites_router import router as favorites_router
from locations_router import router as locations_router
import crud
import schemas

# 새로 만든 model_service에서 필요한 함수들을 import
from model_service import run_prediction, get_model_status, MODEL_LOADED

# --- 로깅, DB 테이블 생성 ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:     %(message)s')
Base.metadata.create_all(bind=engine)

# --- FastAPI 앱 초기화 및 라우터 등록 ---
app = FastAPI(
    title="Midas API - 실내 위치 예측 및 사용자 서비스 (Refactored)",
    description="모델 서비스가 분리되어 구조가 개선된 API 서버입니다.",
    version="9.0.0",
)
app.include_router(user_router, prefix="/users", tags=["users"])
app.include_router(favorites_router, prefix="/favorites", tags=["favorites"])
app.include_router(locations_router, prefix="/locations", tags=["Predicted Locations"])

# --- API Endpoints ---
@app.get("/health", summary="서버 상태 확인")
def health():
    """서버의 현재 동작 상태와 머신러닝 모델의 로드 상태를 확인합니다."""
    # 모델 상태 정보를 model_service에서 가져옵니다.
    return get_model_status()

@app.post("/predict", response_model=schemas.ModelOutput, summary="센서 데이터로 위치 예측")
def predict(data: schemas.SensorInput, db: Session = Depends(get_db)):
    """
    센서 데이터를 받아 위치를 예측하고, DB에서 해당 위치의 상세 정보를 함께 반환합니다.
    항상 상위 3개의 예측 결과를 반환합니다.
    """
    if not MODEL_LOADED:
        return schemas.ModelOutput(
            prediction=-1, confidence=0.0, location_details=None,
            top_k_results=["Error: Model components are not loaded.", "", ""]
        )
    
    # 예측 로직 전체를 model_service의 함수 호출로 대체
    prediction_result, log_results, response_results = run_prediction(data)
    
    # 로깅 처리
    input_str = (
        f"Input: Mag({data.Mag_X:.2f}, {data.Mag_Y:.2f}, {data.Mag_Z:.2f}), "
        f"Ori({data.Ori_X:.2f}, {data.Ori_Y:.2f}, {data.Ori_Z:.2f})"
    )
    if log_results:
        results_str_parts = [f"{i+1}st: {pred} ({conf:.2%})" for i, (pred, conf) in enumerate(log_results)]
        results_str = " | ".join(results_str_parts)
    else:
        results_str = f"Prediction: {prediction_result} (no confidence info)"
    logging.info(f"{input_str} -> {results_str}")
    
    # 응답 데이터 구성
    confidence_score = log_results[0][1] if log_results else None
    
    # 무조건 3개의 결과를 문자열 리스트로 생성
    top_k_list = []
    if response_results and len(response_results) >= 3:
        top_k_list = [f"{pred} ({conf:.4f})" for pred, conf in response_results[:3]]
    else:
        # 결과가 3개 미만인 경우 빈 문자열로 채움
        top_k_list = [f"{pred} ({conf:.4f})" for pred, conf in response_results] if response_results else []
        while len(top_k_list) < 3:
            top_k_list.append("")
    
    # DB에서 위치 상세 정보 조회
    location_info_db = crud.get_predicted_location(db, location_id=prediction_result)
    location_details_schema = schemas.PredictedLocation.model_validate(location_info_db) if location_info_db else None

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
        "version": "9.0.0",
        "model_type": "MLPClassifier with Full Calibration Pipeline (Service-based)",
    }


