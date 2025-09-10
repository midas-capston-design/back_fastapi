from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

import crud
import schemas
# [수정] get_db import 경로를 database.py로 변경하여 일관성을 확보합니다.
from database import get_db

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
    - **id, location_name, description, floor, address** 등 모든 정보를 포함해야 합니다.
    """
    # ID 중복 체크
    db_location_by_id = crud.get_predicted_location(db, location_id=location.id)
    if db_location_by_id:
        raise HTTPException(status_code=400, detail=f"ID {location.id}는 이미 등록된 위치입니다.")
    
    # 수정된 crud 함수를 호출하여 모든 필드를 저장
    return crud.create_predicted_location(db=db, location=location)


@router.get(
    "/",
    response_model=List[schemas.PredictedLocation],
    summary="모든 예측 위치 조회"
)
def get_all_locations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    DB에 저장된 모든 예측 위치 정보 목록을 반환합니다. (페이지네이션 지원)
    """
    # 페이지네이션을 지원하는 crud 함수로 호출
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
    - 변경하려는 필드만 요청 본문에 포함하면 됩니다.
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
    """
    db_location = crud.delete_predicted_location(db, location_id)
    if db_location is None:
        raise HTTPException(status_code=404, detail="해당 ID의 위치를 찾을 수 없습니다.")
    return db_location

