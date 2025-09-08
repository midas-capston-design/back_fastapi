# database.py

# 필요한 import 추가
from sqlalchemy import create_engine, Column, Integer, String, Enum, TIMESTAMP, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum

# --- 데이터베이스 연결 설정 (기존 코드 유지) ---
DB_HOST = "database-2.ch8qsw00uz5o.ap-northeast-2.rds.amazonaws.com"
DB_PORT = 3306
DB_USER = "admin"
DB_PASS = "esc1234!!"
DB_NAME = "user_db"

SQLALCHEMY_DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- SQLAlchemy 엔진 및 세션 생성 (기존 코드 유지) ---
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- 데이터베이스 모델(테이블) 정의 ---

# Favorite 모델에서 사용할 Enum 정의
class FavoriteTypeEnum(enum.Enum):
    place = "place"
    bus = "bus"
    busStop = "busStop"

class PlaceCategoryEnum(enum.Enum):
    home = "home"
    work = "work"
    convenienceStore = "convenienceStore"
    school = "school"
    etc = "etc"


class User(Base):
    # (기존 User 모델 코드 유지)
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    userId = Column(String(50), unique=True, index=True, nullable=False)
    userName = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    phone_number = Column(String(20), nullable=False)
    hashed_password = Column(String(255), nullable=False)

    # User와 Favorite 간의 관계 설정 추가
    favorites = relationship("Favorite", back_populates="owner", cascade="all, delete-orphan")


# Favorite 모델 새로 추가
class Favorite(Base):
    __tablename__ = "favorites"

    id = Column(String(255), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(Enum(FavoriteTypeEnum), nullable=False)
    name = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    address = Column(String(255), nullable=True)
    place_category = Column(Enum(PlaceCategoryEnum), nullable=True)
    bus_number = Column(String(50), nullable=True)
    station_name = Column(String(255), nullable=True)
    station_id = Column(String(255), nullable=True)

    # User와 Favorite 간의 관계 설정 추가
    owner = relationship("User", back_populates="favorites")


# 🔽 PredictedLocation 모델 새로 추가
class PredictedLocation(Base):
    __tablename__ = "predicted_locations"

    id = Column(Integer, primary_key=True)
    location_name = Column(String(255), nullable=False, unique=True)
    description = Column(String(1024), nullable=True) # TEXT 대신 길이 제한이 있는 String 사용도 가능


