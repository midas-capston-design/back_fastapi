# -*- coding: utf-8 -*-
# 사용자 관련 API 엔드포인트를 정의하는 라우터 파일입니다.

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from jose import JWTError, jwt

import crud
import schemas
import auth
import database

router = APIRouter()

# --- 의존성 주입 (Dependency Injection) ---

# 🔽 데이터베이스 세션을 생성하고 반환하는 함수를 여기에 정의합니다.
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        userId: str = payload.get("sub")
        if userId is None:
            raise credentials_exception
        token_data = schemas.TokenData(userId=userId)
    except JWTError:
        raise credentials_exception
    
    user = crud.get_user_by_user_id(db, user_id_str=token_data.userId)
    if user is None:
        raise credentials_exception
    return user

# --- API 엔드포인트 ---

@router.post("/signup", response_model=schemas.User)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # 🔽 호출하는 함수 이름을 'get_user_by_user_id'로 통일합니다.
    db_user = crud.get_user_by_user_id(db, user_id_str=user.userId)
    if db_user:
        raise HTTPException(status_code=400, detail="User ID already registered")
    return crud.create_user(db=db, user=user)

@router.post("/login", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # 🔽 호출하는 함수 이름을 'get_user_by_user_id'로 통일합니다.
    user = crud.get_user_by_user_id(db, user_id_str=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.userId}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=schemas.User)
def read_users_me(current_user: database.User = Depends(get_current_user)):
    return current_user