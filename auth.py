# -*- coding: utf-8 -*-
# 인증(Authentication) 관련 유틸리티 함수들을 모아놓은 파일입니다.

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional

# --- 설정 ---
# ⚠️ 보안을 위해 이 SECRET_KEY는 절대로 코드에 하드코딩하지 마세요.
# 실제 운영 환경에서는 환경 변수 등을 통해 안전하게 관리해야 합니다.
SECRET_KEY = "a_very_secret_key_that_should_be_in_env_vars"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- 비밀번호 처리 ---
# user.py에 있던 것을 이곳으로 이동하여 인증 관련 로직을 통합 관리합니다.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- OAuth2 스키마 ---
# FastAPI가 API 문서에서 인증 UI를 자동으로 생성하고,
# API 요청의 Authorization 헤더에서 토큰을 추출하는 역할을 합니다.
# tokenUrl은 클라이언트가 토큰을 얻기 위해 요청을 보내는 API 경로입니다.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


# --- 함수 ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """입력된 비밀번호와 해시된 비밀번호를 비교합니다."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """비밀번호를 해싱합니다."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    주어진 데이터(payload)를 포함하는 JWT 접근 토큰을 생성합니다.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
