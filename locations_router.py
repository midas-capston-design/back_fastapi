# -*- coding: utf-8 -*-
"""
예측 위치 정보(Predicted Location) 관련 API 라우터

이 파일은 모델이 예측할 수 있는 각 위치(장소)의 상세 정보를
관리하기 위한 API 엔드포인트를 정의합니다. (주로 관리자용)

- POST /: 새 예측 위치 정보 추가
- GET /: 모든 예측 위치 정보 목록 조회
- PUT /{location_id}: 특정 예측 위치 정보 수정
- DELETE /{location_id}: 특정 예측 위치 정보 삭제
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

# --- 프로젝트 내부 모듈 Import ---
import crud
import schemas
from database import get_db

# "/locations" 경로에 대한 API 작업을 그룹화하는 APIRouter 객체를 생성합니다.
router = APIRouter()


@router.post(
    "/",
    response_model=schemas.PredictedLocation,
    status_code=status.HTTP_201_CREATED,
    summary="새 예측 위치 추가"
)
def create_location(
    location: schemas.PredictedLocationCreate,
    db: Session = Depends(get_db)
):
    """
    새로운 예측 위치 정보를 DB에 추가합니다.
    모델의 예측 결과(예: 5)와 실제 장소("3층 로비")를 매핑하기 위해 사용됩니다.

    - **요청**: `schemas.PredictedLocationCreate` 형식의 위치 정보.
    - **응답**: 생성된 위치 정보 (`schemas.PredictedLocation` 형식).
    - **에러**: 해당 `id`가 이미 존재하면 400 (Bad Request) 에러를 반환합니다.
    """
    # DB에 동일한 ID가 있는지 먼저 확인하여 중복 생성을 방지합니다.
    db_location_by_id = crud.get_predicted_location(db, location_id=location.id)
    if db_location_by_id:
        raise HTTPException(status_code=400, detail=f"ID {location.id}는 이미 등록된 위치입니다.")
    
    # 실제 DB 생성 작업은 crud.py의 함수에 위임합니다.
    return crud.create_predicted_location(db=db, location=location)


@router.get(
    "/",
    response_model=List[schemas.PredictedLocation],
    summary="모든 예측 위치 조회"
)
def get_all_locations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    DB에 저장된 모든 예측 위치 정보 목록을 반환합니다. (페이지네이션 지원)

    - **쿼리 파라미터**: `skip`(건너뛸 개수), `limit`(최대 개수)를 통해 페이지네이션이 가능합니다.
    - **응답**: 위치 정보의 리스트 (`List[schemas.PredictedLocation]` 형식).
    """
    locations = crud.get_all_predicted_locations(db=db, skip=skip, limit=limit)
    return locations


@router.put(
    "/{location_id}",
    response_model=schemas.PredictedLocation,
    summary="예측 위치 정보 수정"
)
def update_location(
    location_id: int,
    location_update: schemas.PredictedLocationUpdate,
    db: Session = Depends(get_db)
):
    """
    ID에 해당하는 예측 위치 정보를 수정합니다.

    - **경로 파라미터**: `location_id` (수정할 위치의 고유 ID).
    - **요청**: `schemas.PredictedLocationUpdate` 형식. 변경하려는 필드만 포함하면 됩니다.
    - **응답**: 수정된 위치 정보.
    - **에러**: 해당 ID의 위치가 없으면 404 (Not Found) 에러를 반환합니다.
    """
    db_location = crud.update_predicted_location(db, location_id, location_update)
    if db_location is None:
        raise HTTPException(status_code=404, detail="해당 ID의 위치를 찾을 수 없습니다.")
    return db_location


@router.delete(
    "/{location_id}",
    response_model=schemas.PredictedLocation,
    summary="예측 위치 정보 삭제"
)
def delete_location(
    location_id: int,
    db: Session = Depends(get_db)
):
    """
    ID에 해당하는 예측 위치 정보를 삭제합니다.

    - **경로 파라미터**: `location_id` (삭제할 위치의 고유 ID).
    - **응답**: 삭제된 위치 정보.
    - **에러**: 해당 ID의 위치가 없으면 404 (Not Found) 에러를 반환합니다.
    """
    db_location = crud.delete_predicted_location(db, location_id)
    if db_location is None:
        raise HTTPException(status_code=404, detail="해당 ID의 위치를 찾을 수 없습니다.")
    return db_location

