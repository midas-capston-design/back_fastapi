from sqlalchemy.orm import Session
from fastapi import HTTPException, status
import database, schemas, auth

# --- User CRUD (사용자) ---

def get_user(db: Session, user_id: int):
    """정수 ID로 사용자를 조회합니다."""
    return db.query(database.User).filter(database.User.id == user_id).first()

def get_user_by_user_id(db: Session, user_id_str: str):
    """문자열 userId로 사용자를 조회합니다."""
    return db.query(database.User).filter(database.User.userId == user_id_str).first()

def create_user(db: Session, user: schemas.UserCreate):
    """새로운 사용자를 생성합니다."""
    hashed_password = auth.get_password_hash(user.password)
    db_user = database.User(
        userId=user.userId,
        email=user.email,
        userName=user.userName,
        phone_number=user.phone_number,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Favorite CRUD (즐겨찾기) ---

def get_favorites_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    """특정 사용자의 즐겨찾기 목록을 조회합니다."""
    return db.query(database.Favorite).filter(database.Favorite.user_id == user_id).offset(skip).limit(limit).all()

def create_user_favorite(db: Session, favorite: schemas.FavoriteCreate, user_id: int):
    """사용자에게 새로운 즐겨찾기를 추가합니다."""
    db_existing_favorite = db.query(database.Favorite).filter(database.Favorite.id == favorite.id).first()
    if db_existing_favorite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"ID '{favorite.id}'는 이미 등록된 즐겨찾기입니다."
        )
    # Pydantic V2에 맞춰 .model_dump() 사용
    db_favorite = database.Favorite(**favorite.model_dump(), user_id=user_id)
    db.add(db_favorite)
    db.commit()
    db.refresh(db_favorite)
    return db_favorite

def delete_favorite(db: Session, favorite_id: str, user_id: int):
    """사용자의 즐겨찾기를 삭제합니다."""
    db_favorite = db.query(database.Favorite).filter(database.Favorite.id == favorite_id, database.Favorite.user_id == user_id).first()
    if db_favorite:
        db.delete(db_favorite)
        db.commit()
        return db_favorite
    return None


# --- Predicted Location CRUD (예측 위치) ---

def get_predicted_location(db: Session, location_id: int):
    """ID로 특정 예측 위치 정보를 조회합니다."""
    return db.query(database.PredictedLocation).filter(database.PredictedLocation.id == location_id).first()

def get_all_predicted_locations(db: Session, skip: int = 0, limit: int = 100):
    """모든 예측 위치 정보 목록을 페이지네이션하여 조회합니다."""
    return db.query(database.PredictedLocation).offset(skip).limit(limit).all()

def create_predicted_location(db: Session, location: schemas.PredictedLocationCreate):
    """
    새로운 예측 위치 정보를 생성합니다.
    schemas의 모든 필드(floor, address 포함)를 DB 모델에 매핑합니다.
    """
    db_location = database.PredictedLocation(**location.model_dump())
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location

def update_predicted_location(db: Session, location_id: int, location_update: schemas.PredictedLocationUpdate):
    """ID로 특정 예측 위치 정보를 수정합니다."""
    db_location = get_predicted_location(db, location_id)
    if not db_location:
        return None

    # Pydantic V2에 맞춰 .model_dump() 사용
    update_data = location_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_location, key, value)

    db.commit()
    db.refresh(db_location)
    return db_location

def delete_predicted_location(db: Session, location_id: int):
    """ID로 특정 예측 위치 정보를 삭제합니다."""
    db_location = get_predicted_location(db, location_id)
    if not db_location:
        return None
    
    db.delete(db_location)
    db.commit()
    return db_location

