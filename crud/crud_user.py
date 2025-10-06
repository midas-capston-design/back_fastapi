from sqlalchemy.orm import Session
from typing import Optional
from database import User
from schemas import UserCreate, UserNameUpdate # UserNameUpdate 임포트 추가
from core import security

def get_user_by_user_id(db: Session, user_id_str: str) -> Optional[User]:
    return db.query(User).filter(User.userId == user_id_str).first()

def create_user(db: Session, user: UserCreate) -> User:
    hashed_password = security.get_password_hash(user.password)
    db_user = User(
        **user.model_dump(exclude={"password"}), 
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- 추가된 부분 ---
def update_user_name(db: Session, user_id_str: str, user_update: UserNameUpdate) -> Optional[User]:
    """
    userId로 사용자를 찾아 userName을 변경합니다.
    """
    # 1. DB에서 사용자 찾기
    db_user = get_user_by_user_id(db, user_id_str=user_id_str)

    # 2. 사용자가 있으면 이름 변경 후 저장
    if db_user:
        db_user.userName = user_update.userName
        db.commit()
        db.refresh(db_user)
        return db_user
    
    # 3. 사용자가 없으면 None 반환
    return None