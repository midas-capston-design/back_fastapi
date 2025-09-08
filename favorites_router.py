# -*- coding: utf-8 -*-
# 즐겨찾기 기능 관련 API 엔드포인트를 정의하는 라우터 파일입니다.

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

import crud
import schemas
import database
# 🔽 user_router.py 파일에서 get_current_user와 함께 get_db를 불러옵니다.
from user_router import get_current_user, get_db

router = APIRouter()

# --- API 엔드포인트 정의 ---
# 이제 Depends(get_db)를 호출할 때 get_db 함수를 정상적으로 찾을 수 있습니다.

@router.get("/", response_model=List[schemas.FavoriteResponse], summary="내 모든 즐겨찾기 조회")
def get_favorites(
    db: Session = Depends(get_db), 
    current_user: database.User = Depends(get_current_user)
):
    """
    현재 로그인된 사용자의 모든 즐겨찾기 목록을 반환합니다.
    """
    favorites = crud.get_favorites_by_user(db, user_id=current_user.id)
    return favorites


@router.post("/", response_model=schemas.FavoriteResponse, status_code=status.HTTP_201_CREATED, summary="새 즐겨찾기 추가")
def add_favorite(
    favorite: schemas.FavoriteCreate,
    db: Session = Depends(get_db),
    current_user: database.User = Depends(get_current_user)
):
    """
    요청 본문(body)에 담긴 정보로 새 즐겨찾기를 생성합니다.
    - `type`에 따라 필요한 필드가 다릅니다. (예: type='place'이면 address, category 필수)
    """
    # ID 중복 체크
    db_favorite = db.query(database.Favorite).filter(database.Favorite.id == favorite.id).first()
    if db_favorite:
        raise HTTPException(status_code=400, detail=f"ID '{favorite.id}'는 이미 존재하는 즐겨찾기입니다.")
    
    return crud.create_user_favorite(db=db, favorite=favorite, user_id=current_user.id)


@router.delete("/{favorite_id}", status_code=status.HTTP_204_NO_CONTENT, summary="특정 즐겨찾기 삭제")
def remove_favorite(
    favorite_id: str,
    db: Session = Depends(get_db),
    current_user: database.User = Depends(get_current_user)
):
    """
    URL 경로로 받은 `favorite_id`를 가진 즐겨찾기를 삭제합니다.
    """
    db_favorite = crud.delete_favorite(db=db, favorite_id=favorite_id, user_id=current_user.id)
    if db_favorite is None:
        raise HTTPException(status_code=404, detail="즐겨찾기를 찾을 수 없거나 삭제 권한이 없습니다.")
    
    # 성공 시 본문(body) 없이 204 상태 코드를 반환합니다.
    return