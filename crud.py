

from sqlalchemy.orm import Session
from fastapi import HTTPException, status
import database
import schemas
import auth

# --- User CRUD ---

def get_user(db: Session, user_id: int):
    return db.query(database.User).filter(database.User.id == user_id).first()

# 🔽 이 함수의 이름을 'get_user_by_user_id'로 통일합니다.
def get_user_by_user_id(db: Session, user_id_str: str):
    return db.query(database.User).filter(database.User.userId == user_id_str).first()

def create_user(db: Session, user: schemas.UserCreate):
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

# --- Favorite CRUD ---

def get_favorites_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(database.Favorite).filter(database.Favorite.user_id == user_id).offset(skip).limit(limit).all()

def create_user_favorite(db: Session, favorite: schemas.FavoriteCreate, user_id: int):
    # ⚠️ 2. 중복 체크 시 'item_id'를 올바른 컬럼명 'id'로 수정
    db_existing_favorite = db.query(database.Favorite).filter(
        database.Favorite.id == favorite.id
    ).first()
    
    if db_existing_favorite:
        # 중복 시 에러를 발생시키는 것이 더 명확합니다.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"ID '{favorite.id}'는 이미 등록된 즐겨찾기입니다."
        )

    db_favorite = database.Favorite(**favorite.dict(), user_id=user_id)
    db.add(db_favorite)
    db.commit()
    db.refresh(db_favorite)
    return db_favorite

def delete_favorite(db: Session, favorite_id: str, user_id: int): # ⚠️ 3. 타입을 'str'로 수정
    db_favorite = db.query(database.Favorite).filter(database.Favorite.id == favorite_id, database.Favorite.user_id == user_id).first()
    if db_favorite:
        db.delete(db_favorite)
        db.commit()
        return db_favorite
    return None


# --- Predicted Location CRUD ---

def get_predicted_location(db: Session, location_id: int):
    """ID로 특정 예측 위치 정보를 조회합니다."""
    return db.query(database.PredictedLocation).filter(database.PredictedLocation.id == location_id).first()

def get_all_predicted_locations(db: Session):
    """모든 예측 위치 정보 목록을 조회합니다."""
    return db.query(database.PredictedLocation).all()

def create_predicted_location(db: Session, location: schemas.PredictedLocationCreate):
    """새로운 예측 위치 정보를 생성합니다."""
    db_location = database.PredictedLocation(
        id=location.id,
        location_name=location.location_name,
        description=location.description
    )
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location


def update_predicted_location(db: Session, location_id: int, location_update: schemas.PredictedLocationUpdate):
    """기존 예측 위치 정보를 수정합니다."""
    db_location = get_predicted_location(db, location_id)
    if not db_location:
        return None

    update_data = location_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_location, key, value)

    db.add(db_location)
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





