# -*- coding: utf-8 -*-
"""
즐겨찾기(Favorite) 관련 API 라우터

이 파일은 즐겨찾기 기능과 관련된 API 엔드포인트를 정의합니다.
- POST /: 새 즐겨찾기 추가
- GET /: 내 즐겨찾기 목록 조회
- DELETE /{favorite_id}: 특정 즐겨찾기 삭제

이 라우터의 모든 엔드포인트는 사용자 인증을 필요로 합니다.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

# --- 프로젝트 내부 모듈 Import ---
import crud
import schemas
import core.auth as auth
from database import get_db

# "/favorites" 경로에 대한 API 작업을 그룹화하는 APIRouter 객체를 생성합니다.
router = APIRouter()


@router.post(
    "/",
    response_model=schemas.FavoriteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="새 즐겨찾기 추가 (인증 필요)"
)
def create_favorite_for_user(
    favorite: schemas.FavoriteCreate,
    db: Session = Depends(get_db),
    # `Depends(auth.get_current_active_user)`를 통해 이 엔드포인트가
    # 반드시 인증된 사용자의 요청에만 응답하도록 강제합니다.
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    """
    현재 로그인된 사용자의 새 즐겨찾기를 추가합니다.

    - **인증**: `Authorization: Bearer {토큰}` 헤더가 필요합니다.
    - **요청**: `schemas.FavoriteCreate` 형식의 즐겨찾기 정보.
    - **응답**: 생성된 즐겨찾기 정보 (`schemas.FavoriteResponse` 형식).
    - **에러**: 즐겨찾기 `id`가 이미 존재하면 409 (Conflict) 에러를 반환합니다.
    """
    # 실제 DB 생성 작업은 crud.py의 함수에 위임합니다.
    # 이때, 현재 로그인된 사용자의 id(current_user.id)를 함께 넘겨주어
    # 해당 즐겨찾기의 소유자를 명확히 합니다.
    return crud.create_user_favorite(db=db, favorite=favorite, user_id=current_user.id)


@router.get(
    "/",
    response_model=List[schemas.FavoriteResponse],
    summary="내 즐겨찾기 목록 조회 (인증 필요)"
)
def read_favorites(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    """
    현재 로그인된 사용자의 모든 즐겨찾기 목록을 반환합니다. (페이지네이션 지원)

    - **인증**: `Authorization: Bearer {토큰}` 헤더가 필요합니다.
    - **쿼리 파라미터**: `skip`(건너뛸 개수), `limit`(최대 개수)를 통해 페이지네이션이 가능합니다.
    - **응답**: 즐겨찾기 정보의 리스트 (`List[schemas.FavoriteResponse]` 형식).
    """
    # crud.py의 함수를 호출하여 현재 사용자의 즐겨찾기 목록을 조회합니다.
    favorites = crud.get_favorites_by_user(db, user_id=current_user.id, skip=skip, limit=limit)
    return favorites


@router.delete(
    "/{favorite_id}",
    response_model=schemas.FavoriteResponse,
    summary="즐겨찾기 삭제 (인증 필요)"
)
def delete_favorite(
    favorite_id: str,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    """
    ID에 해당하는 즐겨찾기를 삭제합니다.
    본인의 즐겨찾기만 삭제할 수 있습니다.

    - **인증**: `Authorization: Bearer {토큰}` 헤더가 필요합니다.
    - **경로 파라미터**: `favorite_id` (삭제할 즐겨찾기의 고유 ID).
    - **응답**: 삭제된 즐겨찾기 정보.
    - **에러**: 즐겨찾기가 없거나, 본인의 것이 아닐 경우 404 (Not Found) 에러를 반환합니다.
    """
    # crud.py의 함수를 호출하여 DB에서 즐겨찾기를 삭제합니다.
    # 이때, 사용자 ID를 함께 넘겨주어 다른 사람의 즐겨찾기를 삭제하지 못하도록 합니다.
    db_favorite = crud.delete_favorite(db=db, favorite_id=favorite_id, user_id=current_user.id)
    if db_favorite is None:
        raise HTTPException(status_code=404, detail="해당 ID의 즐겨찾기를 찾을 수 없거나 삭제 권한이 없습니다.")
    
    return db_favorite