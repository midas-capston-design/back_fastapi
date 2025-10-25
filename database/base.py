# -*- coding: utf-8 -*-
"""
데이터베이스 연결 및 세션 관리를 위한 핵심 설정 파일
- SQLAlchemy 엔진, 세션, Base 모델을 정의합니다.
- .env 파일에서 데이터베이스 접속 정보를 로드합니다.
.env 파일은 깃허브에 올리지 않기
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os                     # os 모듈 임포트
from dotenv import load_dotenv # dotenv 임포트

# .env 파일로부터 환경 변수를 로드합니다.
load_dotenv()

# --- 데이터베이스 연결 정보 (환경 변수 사용) ---
# os.getenv("환경변수이름", "기본값") 형식으로 .env 파일에서 값을 읽어옵니다.
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "default_user")
DB_PASS = os.getenv("DB_PASS", "default_password")
DB_NAME = os.getenv("DB_NAME", "default_db")

SQLALCHEMY_DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

# --- SQLAlchemy 핵심 설정 (이하 동일) ---
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

