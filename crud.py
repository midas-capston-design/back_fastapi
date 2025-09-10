# -*- coding: utf-8 -*-
"""
CRUD (Create, Read, Update, Delete) 작업 모듈

이 파일은 데이터베이스와 직접적으로 상호작용하는 모든 함수를 정의합니다.
각 함수는 하나의 특정 작업(예: 사용자 생성, 즐겨찾기 조회)을 담당하며,
API 라우터(예: user_router.py)에서 이 함수들을 호출하여 사용합니다.

CRUD 로직을 별도 파일로 분리하면 API 엔드포인트의 코드를 간결하게 유지하고,
데이터베이스 로직을 재사용하기 용이해집니다.
"""

from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from typing import List

# --- 프로젝트 내부 모듈 Import ---
import database  # 데이터베이스 모델(테이블 클래스)을 가져오기 위해 import
import schemas   # Pydantic 스키마(데이터 형식)를 타입 힌트로 사용하기 위해 import
import security  # 비밀번호 해싱 함수를 사용하기 위해 import


# --- User CRUD (사용자 관련) ---

def get_user_by_user_id(db: Session, user_id_str: str) -> database.User:
    """
    사용자의 로그인 ID(userId)를 기준으로 데이터베이스에서 특정 사용자 정보를 조회합니다.

    Args:
        db (Session): 데이터베이스 세션 객체.
        user_id_str (str): 조회할 사용자의 로그인 ID.

    Returns:
        database.User: 조회된 사용자 객체. 사용자가 없으면 None을 반환합니다.
    """
    return db.query(database.User).filter(database.User.userId == user_id_str).first()

def create_user(db: Session, user: schemas.UserCreate) -> database.User:
    """
    새로운 사용자를 생성하고 데이터베이스에 저장합니다.

    Args:
        db (Session): 데이터베이스 세션 객체.
        user (schemas.UserCreate): 생성할 사용자의 정보가 담긴 Pydantic 스키마.

    Returns:
        database.User: 생성된 사용자 정보 객체.
    """
    # 1. 사용자가 입력한 비밀번호를 security.py의 함수를 이용해 안전하게 해싱합니다.
    hashed_password = security.get_password_hash(user.password)
    
    # 2. Pydantic 모델(user)에서 데이터베이스 모델(database.User)로 데이터를 옮깁니다.
    #    - model_dump()를 사용하되, 평문 비밀번호(password)는 제외합니다.
    #    - 해싱된 비밀번호(hashed_password)를 별도로 추가합니다.
    db_user = database.User(
        **user.model_dump(exclude={"password"}), 
        hashed_password=hashed_password
    )
    
    # 3. 데이터베이스에 사용자 정보를 추가(add), 저장(commit), 최신 정보로 갱신(refresh)합니다.
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# --- Favorite CRUD (즐겨찾기 관련) ---

def get_favorites_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[database.Favorite]:
    """
    특정 사용자의 모든 즐겨찾기 목록을 조회합니다. (페이지네이션 지원)

    Args:
        db (Session): 데이터베이스 세션 객체.
        user_id (int): 즐겨찾기를 조회할 사용자의 고유 ID.
        skip (int): 건너뛸 항목의 수.
        limit (int): 가져올 최대 항목의 수.

    Returns:
        List[database.Favorite]: 조회된 즐겨찾기 객체의 리스트.
    """
    return db.query(database.Favorite).filter(database.Favorite.user_id == user_id).offset(skip).limit(limit).all()

def create_user_favorite(db: Session, favorite: schemas.FavoriteCreate, user_id: int) -> database.Favorite:
    """
    특정 사용자에게 새로운 즐겨찾기를 추가합니다.

    Args:
        db (Session): 데이터베이스 세션 객체.
        favorite (schemas.FavoriteCreate): 생성할 즐겨찾기 정보.
        user_id (int): 즐겨찾기를 추가할 사용자의 고유 ID.

    Returns:
        database.Favorite: 생성된 즐겨찾기 객체.
    """
    # 즐겨찾기 ID가 이미 존재하는지 확인하여 중복 생성을 방지합니다.
    if db.query(database.Favorite).filter(database.Favorite.id == favorite.id).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"ID '{favorite.id}'는 이미 등록된 즐겨찾기입니다."
        )
    
    # Pydantic 모델을 딕셔너리로 변환하여 데이터베이스 모델 객체를 생성합니다.
    db_favorite = database.Favorite(**favorite.model_dump(), user_id=user_id)
    db.add(db_favorite)
    db.commit()
    db.refresh(db_favorite)
    return db_favorite

def delete_favorite(db: Session, favorite_id: str, user_id: int) -> database.Favorite:
    """
    특정 사용자의 특정 즐겨찾기를 삭제합니다.

    Args:
        db (Session): 데이터베이스 세션 객체.
        favorite_id (str): 삭제할 즐겨찾기의 ID.
        user_id (int): 해당 즐겨찾기를 소유한 사용자의 ID.

    Returns:
        database.Favorite: 삭제된 즐겨찾기 객체. 대상이 없으면 None을 반환합니다.
    """
    # 즐겨찾기 ID와 사용자 ID가 모두 일치하는 항목을 찾습니다. (다른 사람의 즐겨찾기 삭제 방지)
    db_favorite = db.query(database.Favorite).filter(database.Favorite.id == favorite_id, database.Favorite.user_id == user_id).first()
    if db_favorite:
        db.delete(db_favorite)
        db.commit()
        return db_favorite
    return None


# --- Predicted Location CRUD (예측 위치 정보 관련) ---

def get_predicted_location(db: Session, location_id: int) -> database.PredictedLocation:
    """ID로 특정 예측 위치의 상세 정보를 조회합니다."""
    return db.query(database.PredictedLocation).filter(database.PredictedLocation.id == location_id).first()

def get_all_predicted_locations(db: Session, skip: int = 0, limit: int = 100) -> List[database.PredictedLocation]:
    """모든 예측 위치 정보 목록을 페이지네이션하여 조회합니다."""
    return db.query(database.PredictedLocation).offset(skip).limit(limit).all()

def create_predicted_location(db: Session, location: schemas.PredictedLocationCreate) -> database.PredictedLocation:
    """새로운 예측 위치 정보를 생성합니다."""
    db_location = database.PredictedLocation(**location.model_dump())
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location

def update_predicted_location(db: Session, location_id: int, location_update: schemas.PredictedLocationUpdate) -> database.PredictedLocation:
    """
    ID로 특정 예측 위치 정보를 수정합니다.
    
    Args:
        location_update (schemas.PredictedLocationUpdate): 수정할 정보. 필드가 비어있으면(None) 해당 필드는 수정하지 않습니다.
    """
    db_location = db.query(database.PredictedLocation).filter(database.PredictedLocation.id == location_id).first()
    if not db_location:
        return None
    
    # Pydantic 모델의 exclude_unset=True 옵션은, 클라이언트가 명시적으로 보낸 값만 업데이트하도록 해줍니다.
    update_data = location_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_location, key, value) # 객체의 속성을 동적으로 설정
    
    db.commit()
    db.refresh(db_location)
    return db_location

def delete_predicted_location(db: Session, location_id: int) -> database.PredictedLocation:
    """ID로 특정 예측 위치 정보를 삭제합니다."""
    db_location = db.query(database.PredictedLocation).filter(database.PredictedLocation.id == location_id).first()
    if db_location:
        db.delete(db_location)
        db.commit()
        return db_location
    return None

