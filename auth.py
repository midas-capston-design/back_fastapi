# -*- coding: utf-8 -*-
"""
인증(Authentication) 및 권한 부여(Authorization) 관련 모듈

이 파일은 사용자 인증과 관련된 모든 기능을 담당합니다.
- 비밀번호 해싱 및 검증
- JWT(JSON Web Token) 생성 및 디코딩
- API 요청 시 사용자를 인증하는 의존성 함수 정의
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

# --- 프로젝트 내부 모듈 Import ---
import crud
import schemas
from database import get_db

# --- 보안 관련 설정 ---

# JWT 서명에 사용할 비밀 키입니다.
# 보안을 위해 실제 운영 환경에서는 반드시 환경 변수로 설정해야 합니다.
# 예: export SECRET_KEY='your_super_secret_key'
SECRET_KEY = os.environ.get("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")

# JWT 생성에 사용할 알고리즘입니다.
ALGORITHM = "HS256"

# JWT 액세스 토큰의 유효 기간(분)입니다.
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- 비밀번호 처리 ---
# bcrypt 알고리즘을 사용하여 비밀번호를 안전하게 해싱합니다.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- OAuth2 스키마 정의 ---
# FastAPI가 API 문서(Swagger UI)에서 인증 UI를 생성하고,
# API 요청의 Authorization 헤더에서 토큰을 추출하는 역할을 합니다.
# tokenUrl은 클라이언트가 토큰을 얻기 위해 요청을 보내는 API 경로입니다.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


# --- 핵심 유틸리티 함수 ---

def verify_password(plain_password, hashed_password):
    """입력된 평문 비밀번호와 데이터베이스의 해시된 비밀번호를 비교합니다."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """평문 비밀번호를 받아 해시 처리하여 반환합니다."""
    return pwd_context.hash(password)

def create_access_token(data: dict):
    """
    JWT 액세스 토큰을 생성합니다.

    Args:
        data (dict): 토큰의 payload에 담을 데이터. 주로 {"sub": user_id} 형태입니다.

    Returns:
        str: 생성된 JWT 토큰 문자열.
    """
    to_encode = data.copy()
    # 토큰 만료 시간을 현재 시간 + 유효 기간으로 설정합니다.
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    # 비밀 키(SECRET_KEY)와 알고리즘을 사용하여 토큰을 서명하고 인코딩합니다.
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(db: Session, username: str, password: str) -> Optional[schemas.User]:
    """
    사용자를 인증합니다. 성공 시 사용자 객체를, 실패 시 None을 반환합니다.

    1. DB에서 사용자 이름(ID)으로 사용자 정보를 조회합니다.
    2. 사용자가 존재하고, 입력된 비밀번호가 DB의 해시된 비밀번호와 일치하는지 확인합니다.
    """
    user = crud.get_user_by_user_id(db, user_id_str=username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def get_current_active_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> schemas.User:
    """
    API 경로 함수에서 'Depends'를 통해 사용되는 의존성 함수입니다.
    요청 헤더의 JWT 토큰을 자동으로 검증하고, 해당 토큰의 사용자 정보를 DB에서 조회하여 반환합니다.
    이 함수 덕분에 각 API 엔드포인트에서는 사용자가 인증되었다고 신뢰하고 비즈니스 로직에 집중할 수 있습니다.

    Args:
        token (str): API 요청의 Authorization 헤더에서 추출된 Bearer 토큰.
        db (Session): 데이터베이스 세션 객체.

    Raises:
        HTTPException(401): 토큰이 유효하지 않거나, 사용자를 찾을 수 없는 경우.

    Returns:
        schemas.User: 인증된 현재 사용자 정보 객체.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="자격 증명을 검증할 수 없습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 토큰을 디코딩하여 payload를 추출합니다.
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # payload에서 사용자 ID("sub" 클레임)를 가져옵니다.
        userId: str = payload.get("sub")
        if userId is None:
            raise credentials_exception
    except JWTError:
        # 토큰 디코딩 과정에서 오류 발생 시 (만료, 서명 불일치 등)
        raise credentials_exception
    
    # 토큰에서 얻은 사용자 ID로 DB에서 실제 사용자 정보를 조회합니다.
    user = crud.get_user_by_user_id(db, user_id_str=userId)
    if user is None:
        # 토큰은 유효하지만 해당 사용자가 DB에 없는 경우
        raise credentials_exception
    
    return user

