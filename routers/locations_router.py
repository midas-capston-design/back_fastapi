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

@router.get(
    "/{location_id}",
    response_model=schemas.PredictedLocation,
    summary="특정 예측 위치 조회"
)
def get_location_by_id(location_id: int, db: Session = Depends(get_db)):
    """
    경로 파라미터로 받은 `location_id`에 해당하는 특정 예측 위치 정보를 조회합니다.
    (예: 지도에서 특정 마커를 클릭했을 때 해당 지점의 상세 정보 요청)

    - **경로 파라미터**: `location_id` (조회할 위치의 고유 ID).
    - **응답**:
        - 성공 시: 해당 ID의 위치 정보 (`schemas.PredictedLocation` 형식).
        - 실패 시: ID에 해당하는 위치가 없으면 **404 Not Found** 에러를 반환합니다.
    """
    db_location = crud.get_predicted_location(db=db, location_id=location_id)
    if db_location is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {location_id}에 해당하는 위치를 찾을 수 없습니다."
        )
    return db_location

