# -*- coding: utf-8 -*-
"""
사용자(User) 관련 API 라우터

이 파일은 사용자 인증 및 관리에 관련된 API 엔드포인트를 정의합니다.
- /signup: 회원가입
- /login: 로그인 (JWT 토큰 발급)
- /me: 내 정보 확인 (인증 필요)

APIRouter를 사용하여 기능별로 API를 그룹화하면, 메인 앱(app.py)의 코드를
체계적으로 관리할 수 있습니다.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

# --- 프로젝트 내부 모듈 Import ---
import crud
import schemas
import auth
from database import get_db

# "/users" 경로에 대한 API 작업을 그룹화하는 APIRouter 객체를 생성합니다.
router = APIRouter()


@router.post("/signup", response_model=schemas.User, status_code=status.HTTP_201_CREATED, summary="회원가입")
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    새로운 사용자를 생성(회원가입)합니다.

    - **요청**: `schemas.UserCreate` 형식의 사용자 정보 (userId, password 등).
    - **응답**: 생성된 사용자 정보 (`schemas.User` 형식, 비밀번호 제외).
    - **에러**: `userId`가 이미 존재하면 400 (Bad Request) 에러를 반환합니다.
    """
    # DB에서 동일한 userId가 있는지 먼저 확인합니다.
    db_user = crud.get_user_by_user_id(db, user_id_str=user.userId)
    if db_user:
        raise HTTPException(status_code=400, detail="이미 등록된 아이디입니다 (userId).")
    
    # 실제 DB 생성 작업은 crud.py의 함수에 위임합니다.
    return crud.create_user(db=db, user=user)


@router.post("/login", response_model=schemas.Token, summary="로그인 및 토큰 발급")
def login_for_access_token(
    db: Session = Depends(get_db),
    # OAuth2PasswordRequestForm: FastAPI가 제공하는 특별한 의존성으로,
    # 클라이언트가 'form-data' 형식으로 보낸 'username'과 'password'를 자동으로 파싱해줍니다.
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    사용자 아이디(username)와 비밀번호(password)로 로그인하여 JWT 액세스 토큰을 발급받습니다.

    - **요청**: Form 데이터 형식의 `username`과 `password`.
    - **응답**: `schemas.Token` 형식의 액세스 토큰.
    - **에러**: 인증에 실패하면 401 (Unauthorized) 에러를 반환합니다.
    """
    # 사용자 이름과 비밀번호가 유효한지 확인하는 인증 로직은 auth.py의 함수에 위임합니다.
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
            headers={"WWW-Authenticate": "Bearer"}, # 응답 헤더에 인증 방식을 명시
        )
    
    # 인증에 성공하면, 해당 사용자의 ID(sub)를 담은 JWT 토큰을 생성합니다.
    access_token = auth.create_access_token(
        data={"sub": user.userId}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=schemas.User, summary="내 정보 확인 (인증 필요)")
def read_users_me(current_user: schemas.User = Depends(auth.get_current_active_user)):
    """
    현재 로그인된 사용자의 정보를 반환합니다.
    이 엔드포인트는 인증된 사용자만 접근할 수 있습니다.

    - **인증**: 요청 헤더에 `Authorization: Bearer {토큰}`이 포함되어야 합니다.
      `Depends(auth.get_current_active_user)` 코드가 토큰을 자동으로 검증하고,
      유효하지 않으면 401 에러를 발생시킵니다.
    - **응답**: 인증된 사용자의 정보 (`schemas.User` 형식).
    """
    # auth.get_current_active_user 함수가 성공적으로 사용자 객체를 반환하면,
    # 그 객체를 그대로 클라이언트에게 응답으로 보냅니다.
    return current_user

