# -*- coding: utf-8 -*-
"""
데이터베이스 설정 및 모델 정의 모듈

이 파일은 프로젝트의 모든 데이터베이스 관련 설정을 담당합니다.
- 데이터베이스 연결을 위한 엔진(engine) 생성
- 데이터베이스 세션(Session) 관리
- SQLAlchemy ORM(Object Relational Mapper)을 위한 기본 모델(Base) 정의
- API가 사용할 데이터베이스 테이블(User, Favorite 등)을 클래스 형태로 정의
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, Enum as SAEnum, ForeignKey, TIMESTAMP
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum

# --- 데이터베이스 연결 설정 ---
# AWS RDS 또는 로컬 데이터베이스의 접속 정보를 설정합니다.
DB_HOST = "database-2.ch8qsw00uz5o.ap-northeast-2.rds.amazonaws.com"
DB_USER = "admin"
DB_PASS = "esc1234!!"
DB_NAME = "user_db"

# SQLAlchemy 연결 URL 형식에 맞게 문자열을 생성합니다.
# 형식: "{DB타입}+{드라이버}://{사용자이름}:{비밀번호}@{호스트주소}/{DB이름}"
SQLALCHEMY_DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"


# --- SQLAlchemy 핵심 설정 ---

# create_engine: 데이터베이스와 통신하는 시작점입니다.
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# sessionmaker: 데이터베이스와의 대화(세션)를 위한 클래스를 생성합니다.
# autocommit=False: 데이터를 변경해도 자동으로 DB에 반영되지 않도록 설정 (db.commit()을 해야 반영됨)
# autoflush=False: 세션이 자동으로 flush(DB에 임시 반영)되지 않도록 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# declarative_base: 모든 모델 클래스가 상속받을 기본 클래스입니다.
# 이 클래스를 상속받는 모든 클래스는 SQLAlchemy에 의해 테이블로 관리됩니다.
Base = declarative_base()


def get_db():
    """
    FastAPI의 의존성 주입(Dependency Injection)을 위한 데이터베이스 세션 생성 함수입니다.

    API 요청이 들어올 때마다 이 함수가 호출되어 독립적인 DB 세션을 생성하고,
    요청 처리가 끝나면 세션을 자동으로 닫아줍니다.
    이를 통해 API 간에 DB 연결이 섞이지 않고 안전하게 관리됩니다.
    """
    db = SessionLocal()
    try:
        yield db # API 함수에 db 세션을 전달합니다.
    finally:
        db.close() # API 처리가 끝나면 세션을 닫습니다.


# --- 데이터베이스에서 사용할 Enum 타입 정의 ---
# Python의 Enum을 SQLAlchemy 모델에서 사용할 수 있도록 정의합니다.

class FavoriteTypeEnum(enum.Enum):
    """즐겨찾기 종류를 나타내는 Enum"""
    place = "place"       # 장소
    bus = "bus"           # 버스
    busStop = "busStop"   # 버스 정류장

class PlaceCategoryEnum(enum.Enum):
    """장소 즐겨찾기의 카테고리를 나타내는 Enum"""
    home = "home"
    work = "work"
    convenienceStore = "convenienceStore"
    school = "school"
    etc = "etc"


# --- 데이터베이스 모델(테이블) 정의 ---
# Base 클래스를 상속받아 데이터베이스 테이블을 파이썬 클래스로 정의합니다.

class User(Base):
    """사용자 정보 테이블 (users)"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True) # 사용자 고유 번호 (자동 증가)
    userId = Column(String(50), unique=True, index=True, nullable=False) # 사용자 로그인 ID
    userName = Column(String(50), nullable=False) # 사용자 이름
    email = Column(String(100), unique=True, index=True, nullable=False) # 이메일
    phone_number = Column(String(20), nullable=False) # 전화번호
    hashed_password = Column(String(255), nullable=False) # 해싱된 비밀번호

    # 다른 테이블과의 관계(Relationship) 설정
    # "Favorite" 모델과 1:N 관계를 맺습니다. (한 명의 사용자는 여러 개의 즐겨찾기를 가질 수 있음)
    # back_populates="owner": Favorite 모델의 'owner' 속성과 연결됨을 의미합니다.
    # cascade="all, delete-orphan": 사용자가 삭제되면, 해당 사용자의 모든 즐겨찾기도 함께 삭제됩니다.
    favorites = relationship("Favorite", back_populates="owner", cascade="all, delete-orphan")

class Favorite(Base):
    """즐겨찾기 정보 테이블 (favorites)"""
    __tablename__ = "favorites"

    id = Column(String(255), primary_key=True) # 즐겨찾기 고유 ID (클라이언트에서 생성)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # 이 즐겨찾기를 소유한 사용자의 ID (외래 키)
    type = Column(SAEnum(FavoriteTypeEnum), nullable=False) # 즐겨찾기 종류 (place, bus, busStop)
    name = Column(String(255), nullable=False) # 즐겨찾기 이름
    created_at = Column(TIMESTAMP, server_default=func.now()) # 생성 일시 (서버에서 자동 생성)
    
    # 장소(place) 타입일 때 사용되는 컬럼들
    address = Column(String(255), nullable=True) # 주소
    place_category = Column(SAEnum(PlaceCategoryEnum), nullable=True) # 장소 카테고리

    # 버스(bus) 또는 버스정류장(busStop) 타입일 때 사용되는 컬럼들
    bus_number = Column(String(50), nullable=True) # 버스 번호
    station_name = Column(String(255), nullable=True) # 정류장 이름
    station_id = Column(String(255), nullable=True) # 정류장 ID

    # "User" 모델과의 관계 설정
    # back_populates="favorites": User 모델의 'favorites' 속성과 연결됨을 의미합니다.
    owner = relationship("User", back_populates="favorites")

class PredictedLocation(Base):
    """모델이 예측하는 위치(장소)에 대한 상세 정보 테이블 (predicted_locations)"""
    __tablename__ = "predicted_locations"

    id = Column(Integer, primary_key=True) # 위치 고유 번호 (모델의 예측 결과와 일치)
    location_name = Column(String(255), nullable=True, unique=True) # 위치 이름 (예: "3층 로비")
    description = Column(Text, nullable=True) # 위치에 대한 상세 설명
    floor = Column(Integer, nullable=False, default=3) # 층수
    address = Column(String(255), nullable=False, default="영남대학교 IT관") # 주소

