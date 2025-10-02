from sqlalchemy.orm import Session
from database import User  # database/__init__.py 덕분에 가능
from schemas import UserCreate # schemas/__init__.py 덕분에 가능
from core import security

def get_user_by_user_id(db: Session, user_id_str: str) -> User:
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

