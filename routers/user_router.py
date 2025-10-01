from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
import crud
import schemas
from core import auth
from database import get_db

router = APIRouter()

@router.post("/signup", response_model=schemas.User, status_code=status.HTTP_201_CREATED, summary="회원가입")
def create_new_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_user_id(db, user_id_str=user.userId)
    if db_user:
        raise HTTPException(status_code=400, detail="이미 등록된 아이디입니다 (userId).")
    return crud.create_user(db=db, user=user)

@router.post("/login", response_model=schemas.Token, summary="로그인 및 토큰 발급")
def login_for_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.userId})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=schemas.User, summary="내 정보 확인 (인증 필요)")
def read_users_me(current_user: schemas.User = Depends(auth.get_current_active_user)):
    return current_user
