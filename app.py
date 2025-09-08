# -*- coding: utf-8 -*-
"""
FastAPI 서버: 위치 기반 자기장 보정 + PyTorch MLP 위치 예측 + 사용자 관리 + 예측 위치 정보
필요 파일:
  - mlp_position.pt
  - label_encoder.pkl
  - scaler.pkl
  - calibration_params.json
"""

import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from scipy.spatial.distance import euclidean
from typing import Dict, List, Optional

# --- 라우터 및 DB import ---
from database import Base, engine
from user_router import router as user_router, get_db
from favorites_router import router as favorites_router
from locations_router import router as locations_router # 새로 만든 라우터 import

import crud # crud 모듈 import
import schemas # schemas 모듈 import

# Create database tables on startup
Base.metadata.create_all(bind=engine)
# ----------------------------------

# -------------------------------
# 설정
# -------------------------------
MODEL_PTH = "model/mlp_position.pt"
LE_PKL = "model/label_encoder.pkl"
SCALER_PKL = "model/scaler.pkl"
CALIB_JSON = "model/calibration_params.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 데이터 입력 스키마
# -------------------------------
class SensorInput(BaseModel):
    Mag_X: float
    Mag_Y: float
    Mag_Z: float
    Ori_X: float  # Azimuth
    Ori_Y: float  # Pitch
    Ori_Z: float  # Roll

# ModelOutput 스키마 수정
class ModelOutput(BaseModel):
    prediction: int
    confidence: float
    calibration_info: dict
    location_details: Optional[schemas.PredictedLocation] = None # 예측 위치 정보 필드 추가

# -------------------------------
# 모델 정의 (학습 코드와 동일해야 함)
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------
# 위치 기반 캘리브레이션 매니저
# -------------------------------
class LocationBasedCalibrationManager:
    def __init__(self, calib_params: Dict):
        self.calib_params = calib_params
        self.best_quality_point = self._find_best_quality_point()
        
    def _find_best_quality_point(self) -> str:
        """최고 품질 캘리브레이션 포인트 (백업용)"""
        best_quality = 0
        best_key = "1"
        
        for key, params in self.calib_params.items():
            if params["calibration_quality"] > best_quality:
                best_quality = params["calibration_quality"]
                best_key = key
        
        return best_key
    
    def find_best_matching_location(self, mag_raw: List[float]) -> str:
        """현재 센서 상태와 가장 유사한 위치 찾기"""
        best_score = float('-inf')
        best_location = self.best_quality_point
        mag_array = np.array(mag_raw)
        
        for point_id, params in self.calib_params.items():
            score = self._calculate_location_score(mag_array, params, point_id)
            if score > best_score:
                best_score = score
                best_location = point_id
                
        return best_location
    
    def _calculate_location_score(self, mag_raw: np.ndarray, params: Dict, point_id: str) -> float:
        """위치별 점수 계산"""
        score = 0
        quality_score = params["calibration_quality"] * 0.3
        score += quality_score
        
        bias = np.array(params["hard_iron_bias"])
        matrix = np.array(params["soft_iron_matrix"])
        
        try:
            corrected = mag_raw - bias
            calibrated = matrix @ corrected
            calibrated_magnitude = np.linalg.norm(calibrated)
            
            if 20 <= calibrated_magnitude <= 80:
                magnitude_error = abs(calibrated_magnitude - 45) / 25
                magnitude_score = max(0, 1.0 - magnitude_error) * 0.4
                score += magnitude_score
            else:
                score -= 0.2
                
        except Exception:
            score -= 0.3
        
        try:
            bias_distance = euclidean(mag_raw, bias)
            max_distance = 20.0
            similarity = max(0, 1.0 - bias_distance / max_distance)
            similarity_score = similarity * 0.3
            score += similarity_score
            
        except Exception:
            score -= 0.1
        
        return score
    
    def apply_location_based_calibration(self, mag_raw: List[float]) -> Dict:
        """위치 기반 적응형 캘리브레이션 적용"""
        best_location = self.find_best_matching_location(mag_raw)
        params = self.calib_params[best_location]
        bias = np.array(params["hard_iron_bias"])
        matrix = np.array(params["soft_iron_matrix"])
        
        try:
            corrected = np.array(mag_raw) - bias
            calibrated = matrix @ corrected
            calibrated_magnitude = np.linalg.norm(calibrated)
            is_reasonable = 15 <= calibrated_magnitude <= 100
            
            if not is_reasonable:
                raise ValueError(f"unreasonable_magnitude_{calibrated_magnitude:.2f}")
            
            return {
                "calibrated_mag": calibrated.tolist(),
                "method": "location_based",
                "selected_location": best_location,
                "calibrated_magnitude": float(calibrated_magnitude),
                "quality": params["calibration_quality"],
                "is_fallback": False
            }
            
        except Exception as e:
            fallback_params = self.calib_params[self.best_quality_point]
            fallback_bias = np.array(fallback_params["hard_iron_bias"])
            fallback_matrix = np.array(fallback_params["soft_iron_matrix"])
            
            fallback_corrected = np.array(mag_raw) - fallback_bias
            fallback_calibrated = fallback_matrix @ fallback_corrected
            
            return {
                "calibrated_mag": fallback_calibrated.tolist(),
                "method": "location_based_with_fallback",
                "selected_location": best_location,
                "fallback_location": self.best_quality_point,
                "calibrated_magnitude": float(np.linalg.norm(fallback_calibrated)),
                "quality": fallback_params["calibration_quality"],
                "is_fallback": True,
                "fallback_reason": str(e)
            }

# -------------------------------
# 캘리브레이션 파라미터 및 매니저 초기화
# -------------------------------
with open(CALIB_JSON, "r", encoding="utf-8") as f:
    calib_params = json.load(f)

calib_manager = LocationBasedCalibrationManager(calib_params)

# -------------------------------
# 전처리 함수 (특징 생성)
# -------------------------------
def build_features(mag, ori):
    B = np.array(mag, dtype=float).reshape(1, -1)
    B_mag = np.linalg.norm(B, axis=1, keepdims=True)
    B_xy = np.linalg.norm(B[:, :2], axis=1, keepdims=True)
    eps = 1e-9
    B_unit = B / (B_mag + eps)

    feat = {}
    feat["B_x"], feat["B_y"], feat["B_z"] = B[0]
    feat["B_mag"] = B_mag[0, 0]
    feat["B_xy_mag"] = B_xy[0, 0]
    feat["Bux"], feat["Buy"], feat["Buz"] = B_unit[0]

    for i, a in enumerate(["Ori_X", "Ori_Y", "Ori_Z"]):
        rad = np.deg2rad(ori[i])
        feat[f"{a}_sin"] = np.sin(rad)
        feat[f"{a}_cos"] = np.cos(rad)

    return np.array(list(feat.values()), dtype=float).reshape(1, -1)

# -------------------------------
# 모델, 인코더, 스케일러 로드
# -------------------------------
le = joblib.load(LE_PKL)
scaler = joblib.load(SCALER_PKL)

input_dim = len(scaler.mean_)
output_dim = len(le.classes_)

model = MLP(input_dim, [2048, 1024, 512], output_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
model.eval()

# -------------------------------
# FastAPI 앱
# -------------------------------
app = FastAPI(title="Location-based Magnetometer Calibration + MLP Inference + User Management")

# --- 라우터 등록 ---
app.include_router(user_router, prefix="/users", tags=["users"])
app.include_router(favorites_router, prefix="/favorites", tags=["favorites"])
app.include_router(locations_router, prefix="/locations", tags=["Predicted Locations"]) # 새로 만든 라우터 등록

# ------------------------------

@app.post("/predict", response_model=ModelOutput)
def predict(data: SensorInput, db: Session = Depends(get_db)): # db 세션 의존성 추가
    """위치 기반 적응형 캘리브레이션을 사용한 위치 예측 및 상세 정보 반환"""
    
    # 1. 위치 기반 자기장 보정
    mag_raw = [data.Mag_X, data.Mag_Y, data.Mag_Z]
    calib_result = calib_manager.apply_location_based_calibration(mag_raw)
    mag_calibrated = calib_result["calibrated_mag"]

    # 2. 특징 생성
    ori_raw = [data.Ori_X, data.Ori_Y, data.Ori_Z]
    feat = build_features(mag_calibrated, ori_raw)

    # 3. 표준화
    feat_std = scaler.transform(feat)

    # 4. 모델 추론
    with torch.no_grad():
        x = torch.tensor(feat_std, dtype=torch.float32).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # 5. 예측된 인덱스(ID)로 DB에서 위치 정보 조회 (새로 추가된 로직)
    location_info = crud.get_predicted_location(db, location_id=pred_idx)

    # 6. 최종 응답에 위치 정보 포함하여 반환 (수정된 로직)
    return ModelOutput(
        prediction=pred_idx, 
        confidence=confidence,
        calibration_info=calib_result,
        location_details=location_info # 위치 정보 추가
    )

# --- 추가된 캘리브레이션 관련 엔드포인트 ---

@app.get("/calibration/stats")
def get_calibration_stats():
    """캘리브레이션 통계 정보 제공"""
    qualities = [params["calibration_quality"] for params in calib_params.values()]
    
    return {
        "total_locations": len(calib_params),
        "quality_stats": {
            "mean": float(np.mean(qualities)),
            "std": float(np.std(qualities)),
            "min": float(np.min(qualities)),
            "max": float(np.max(qualities))
        },
        "best_quality_location": calib_manager.best_quality_point,
        "best_quality_value": calib_params[calib_manager.best_quality_point]["calibration_quality"]
    }

@app.post("/calibration/test")
def test_calibration_selection(data: SensorInput):
    """특정 센서 데이터에 대한 캘리브레이션 선택 과정 상세 정보"""
    mag_raw = [data.Mag_X, data.Mag_Y, data.Mag_Z]
    scores = {}
    mag_array = np.array(mag_raw)
    
    for point_id, params in calib_params.items():
        score = calib_manager._calculate_location_score(mag_array, params, point_id)
        scores[point_id] = {
            "score": float(score),
            "quality": params["calibration_quality"]
        }
    
    best_location = max(scores.keys(), key=lambda k: scores[k]["score"])
    calib_result = calib_manager.apply_location_based_calibration(mag_raw)
    
    return {
        "input_magnetometer": mag_raw,
        "all_location_scores": scores,
        "selected_location": best_location,
        "calibration_result": calib_result,
        "selection_reasoning": {
            "total_locations_evaluated": len(scores),
            "score_range": {
                "min": min(s["score"] for s in scores.values()),
                "max": max(s["score"] for s in scores.values())
            }
        }
    }

@app.get("/")
def root():
    return {
        "message": "Location-based Magnetometer Calibration API with User Management",
        "version": "2.1",
        "features": [
            "29개 위치별 적응형 캘리브레이션",
            "실시간 최적 위치 선택",
            "백업 캘리브레이션 시스템", 
            "MLP 기반 위치 예측",
            "예측 위치 상세 정보 반환",
            "사용자 관리 시스템 (회원가입, 로그인)",
            "즐겨찾기 관리 시스템 (추가, 조회, 삭제)",
            "예측 위치 정보 관리 시스템"
        ],
        "endpoints": {
            "prediction": "/predict",
            "user_management": "/users/*",
            "favorites_management": "/favorites/*",
            "locations_management": "/locations/*",
            "calibration_stats": "/calibration/stats",
            "calibration_test": "/calibration/test"
        }
    }
