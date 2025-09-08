# -*- coding: utf-8 -*-
# 사용자 모델 및 데이터베이스 관련 기능을 담당하는 파일입니다.

from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import database as db
import auth  # auth.py 파일 임포트

# --- Pydantic 모델 ---

class UserCreate(BaseModel):
    """
    회원가입 시 클라이언트로부터 받는 사용자 정보 모델입니다.
    """
    userId: str
    userName: str
    email: EmailStr
    phone_number: str
    password: str

class User(BaseModel):
    """
    API 응답으로 클라이언트에게 반환될 사용자 정보 모델입니다.
    """
    id: int
    userId: str
    userName: str
    email: EmailStr
    phone_number: str

    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    """
    로그인 시 클라이언트로부터 받는 사용자 정보 모델입니다.
    """
    userId: str
    password: str

class Token(BaseModel):
    """
    로그인 성공 시 클라이언트에게 반환될 JWT 토큰 모델입니다.
    """
    access_token: str
    token_type: str


# --- 데이터베이스 CRUD 함수 ---

def get_user(db_session: Session, userId: str):
    """
    사용자 아이디로 데이터베이스에서 사용자를 조회합니다.
    """
    return db_session.query(db.User).filter(db.User.userId == userId).first()

def get_user_by_email(db_session: Session, email: str):
    """
    이메일로 데이터베이스에서 사용자를 조회합니다.
    """
    return db_session.query(db.User).filter(db.User.email == email).first()

def create_user(db_session: Session, user: UserCreate):
    """
    새로운 사용자를 생성하고 데이터베이스에 저장합니다.
    """
    hashed_password = auth.get_password_hash(user.password)
    db_user = db.User(
        userId=user.userId,
        userName=user.userName,
        email=user.email,
        phone_number=user.phone_number,
        hashed_password=hashed_password
    )
    db_session.add(db_user)
    db_session.commit()
    db_session.refresh(db_user)
    return db_user

def authenticate_user(db_session: Session, userId: str, password: str):
    """
    사용자 아이디와 비밀번호를 검증합니다.
    성공 시 사용자 객체를, 실패 시 None을 반환합니다.
    """
    user = get_user(db_session, userId=userId)
    if not user:
        return None
    if not auth.verify_password(password, user.hashed_password):
        return None
    return user