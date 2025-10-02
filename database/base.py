# -*- coding: utf-8 -*-
"""
데이터베이스 연결 및 세션 관리를 위한 핵심 설정 파일
- SQLAlchemy 엔진, 세션, Base 모델을 정의합니다.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# --- 데이터베이스 연결 정보 ---
DB_HOST = "database-2.ch8qsw00uz5o.ap-northeast-2.rds.amazonaws.com"
DB_USER = "admin"
DB_PASS = "esc1234!!"
DB_NAME = "user_db"

SQLALCHEMY_DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

# --- SQLAlchemy 핵심 설정 ---
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI 의존성 주입을 위한 DB 세션 생성 함수"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
