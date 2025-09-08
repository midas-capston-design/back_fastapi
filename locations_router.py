# locations_router.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

import crud
import schemas
from user_router import get_db

router = APIRouter()

@router.post("/", response_model=schemas.PredictedLocation, status_code=status.HTTP_201_CREATED, summary="새 예측 위치 추가")
def create_location(
    location: schemas.PredictedLocationCreate,
    db: Session = Depends(get_db)
):
    """
    새로운 예측 위치 정보를 DB에 추가합니다. (주로 관리자용 기능)
    - id: 모델의 예측 결과 index
    - location_name: 사람이 읽을 수 있는 위치 이름
    """
    db_location_by_id = crud.get_predicted_location(db, location_id=location.id)
    if db_location_by_id:
        raise HTTPException(status_code=400, detail=f"ID {location.id}는 이미 등록된 위치입니다.")
    
    # location_name 중복 체크도 추가할 수 있습니다.
    
    return crud.create_predicted_location(db=db, location=location)


@router.get("/", response_model=List[schemas.PredictedLocation], summary="모든 예측 위치 조회")
def get_all_locations(db: Session = Depends(get_db)):
    """
    DB에 저장된 모든 예측 위치 정보 목록을 반환합니다.
    """
    return crud.get_all_predicted_locations(db=db)


@router.put("/{location_id}", response_model=schemas.PredictedLocation, summary="예측 위치 정보 수정")
def update_location(
    location_id: int,
    location_update: schemas.PredictedLocationUpdate,
    db: Session = Depends(get_db)
):
    """
    ID에 해당하는 예측 위치 정보를 수정합니다.
    - location_name: 새로운 위치 이름
    - description: 새로운 위치 설명
    """
    db_location = crud.update_predicted_location(db, location_id, location_update)
    if db_location is None:
        raise HTTPException(status_code=404, detail="해당 ID의 위치를 찾을 수 없습니다.")
    return db_location

@router.delete("/{location_id}", response_model=schemas.PredictedLocation, summary="예측 위치 정보 삭제")
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
