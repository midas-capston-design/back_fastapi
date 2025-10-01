from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from typing import List
from database import Favorite
from schemas import FavoriteCreate

def get_favorites_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Favorite]:
    return db.query(Favorite).filter(Favorite.user_id == user_id).offset(skip).limit(limit).all()

def create_user_favorite(db: Session, favorite: FavoriteCreate, user_id: int) -> Favorite:
    if db.query(Favorite).filter(Favorite.id == favorite.id).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"ID '{favorite.id}'는 이미 등록된 즐겨찾기입니다."
        )
    db_favorite = Favorite(**favorite.model_dump(), user_id=user_id)
    db.add(db_favorite)
    db.commit()
    db.refresh(db_favorite)
    return db_favorite

def delete_favorite(db: Session, favorite_id: str, user_id: int) -> Favorite:
    db_favorite = db.query(Favorite).filter(Favorite.id == favorite_id, Favorite.user_id == user_id).first()
    if db_favorite:
        db.delete(db_favorite)
        db.commit()
        return db_favorite
    return None
