# -*- coding: utf-8 -*-
"""
사용자 인증(Authentication) 및 JWT 관련 기능을 담당하는 모듈
- JWT 토큰 생성 및 검증
- 비밀번호 확인 및 사용자 인증
- API 요청 시 현재 사용자 정보를 가져오는 의존성 함수 제공
"""

import os                     # os 모듈 임포트
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
# from dotenv import load_dotenv # database/base.py에서 이미 로드하므로 여기서 생략 가능

import schemas
import crud
from database import get_db, User
from . import security # 같은 core 폴더 내의 security 모듈 임포트

# --- 환경 변수에서 SECRET_KEY 읽어오기 ---
# os.getenv("환경변수이름", "기본값") 형식으로 .env 파일에서 값을 읽어옵니다.
# .env 파일에 SECRET_KEY가 없을 경우를 대비해 안전한 기본값을 설정하는 것이 좋습니다.
SECRET_KEY = os.getenv("SECRET_KEY", "a_secure_default_secret_key_if_env_is_missing")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# OAuth2 스키마 정의 (토큰을 얻는 URL 지정)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")

# --- 핵심 유틸리티 함수 ---

def create_access_token(data: dict) -> str:
    """
    JWT 액세스 토큰을 생성합니다.
    Args:
        data (dict): 토큰 payload에 담을 데이터 (주로 {"sub": user_id})
    Returns:
        str: 생성된 JWT 토큰 문자열
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    # 수정된 SECRET_KEY 변수를 사용하여 토큰을 인코딩합니다.
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    사용자 이름과 비밀번호를 검증하여 사용자를 인증합니다.
    성공 시 사용자 객체를, 실패 시 None을 반환합니다.
    """
    user = crud.get_user_by_user_id(db, user_id_str=username)
    # security 모듈의 비밀번호 검증 함수 사용
    if not user or not security.verify_password(password, user.hashed_password):
        return None
    return user

def get_current_active_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    API 경로 함수에서 사용되는 의존성 함수입니다.
    요청 헤더의 JWT 토큰을 검증하고, 해당 사용자의 정보를 DB에서 조회하여 반환합니다.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="자격 증명을 검증할 수 없습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 수정된 SECRET_KEY 변수를 사용하여 토큰을 디코딩합니다.
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        userId: Optional[str] = payload.get("sub")
        if userId is None:
            raise credentials_exception
        # 토큰 데이터 스키마를 사용하여 유효성 검증 (선택 사항이지만 좋은 습관)
        token_data = schemas.TokenData(userId=userId)
    except JWTError:
        raise credentials_exception
    
    # 토큰에서 얻은 userId로 DB에서 실제 사용자 정보를 조회합니다.
    user = crud.get_user_by_user_id(db, user_id_str=token_data.userId)
    if user is None:
        # 토큰은 유효하지만 해당 사용자가 DB에 없는 경우
        raise credentials_exception
    
    # 현재는 별도의 '활성' 상태 체크 로직이 없으므로 바로 사용자 객체 반환
    return user

