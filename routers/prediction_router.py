# -*- coding: utf-8 -*-
from fastapi import APIRouter
import schemas
from services import run_prediction, MODEL_LOADED

# 새로운 APIRouter 객체를 생성합니다.
router = APIRouter()

# 데코레이터를 @app.post에서 @router.post로 변경합니다.
@router.post("/predict", response_model=schemas.ModelOutput, summary="센서 데이터로 위치 예측")
def predict(data: schemas.SensorInput):
    """
    센서 데이터를 받아 위치를 예측합니다.
    """
    if not MODEL_LOADED:
        return schemas.ModelOutput(
            prediction=-1, confidence=0.0,
            top_k_results=["Error: Model components are not loaded.", "", ""]
        )

    prediction_result, log_results, response_results = run_prediction(data)
    
    # 로깅 및 응답 구성 로직은 기존과 동일합니다.
    input_str = (
        f"Input: Mag({data.Mag_X:.2f}, {data.Mag_Y:.2f}, {data.Mag_Z:.2f}), "
        f"Ori({data.Ori_X:.2f}, {data.Ori_Y:.2f}, {data.Ori_Z:.2f})"
    )
    results_str = " | ".join([f"{i+1}st: {p} ({c:.2%})" for i, (p, c) in enumerate(log_results)]) if log_results else f"Prediction: {prediction_result}"
    
    # logging.info()는 app.py에서 전역으로 설정했으므로 여기서 바로 사용 가능합니다.
    # import logging을 추가해도 좋습니다.
    import logging
    logging.info(f"{input_str} -> {results_str}")

    confidence_score = log_results[0][1] if log_results else None
    top_k_list = [f"{p} ({c:.4f})" for p, c in response_results] if response_results else []
    while len(top_k_list) < 3:
        top_k_list.append("")

    return schemas.ModelOutput(
        prediction=prediction_result,
        confidence=confidence_score,
        top_k_results=top_k_list
    )