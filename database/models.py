# -*- coding: utf-8 -*-
"""
데이터베이스 테이블과 매핑되는 SQLAlchemy 모델(ORM 클래스)을 정의하는 파일
"""
import enum
from sqlalchemy import (
    Column, Integer, String, Text, Enum as SAEnum, 
    ForeignKey, TIMESTAMP
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base  # base.py에서 정의한 Base를 가져옵니다.
from sqlalchemy import Boolean, Double

# --- Enum 타입 정의 ---
class FavoriteTypeEnum(enum.Enum):
    place = "place"; bus = "bus"; busStop = "busStop"

class PlaceCategoryEnum(enum.Enum):
    home = "home"; work = "work"; convenienceStore = "convenienceStore"
    school = "school"; etc = "etc"

# --- 데이터베이스 모델(테이블) 정의 ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    userId = Column(String(50), unique=True, index=True, nullable=False)
    userName = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    phone_number = Column(String(20), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    favorites = relationship("Favorite", back_populates="owner", cascade="all, delete-orphan")

class Favorite(Base):
    __tablename__ = "favorites"
    id = Column(String(255), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(SAEnum(FavoriteTypeEnum), nullable=False)
    name = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    address = Column(String(255), nullable=True)
    place_category = Column(SAEnum(PlaceCategoryEnum), nullable=True)
    bus_number = Column(String(50), nullable=True)
    station_name = Column(String(255), nullable=True)
    station_id = Column(String(255), nullable=True)
    owner = relationship("User", back_populates="favorites")

class PredictedLocation(Base):
    __tablename__ = "predicted_locations"
    id = Column(Integer, primary_key=True)
    location_name = Column(String(255), nullable=True, unique=True)
    description = Column(Text, nullable=True)
    floor = Column(Integer, nullable=False, default=3)
    address = Column(String(255), nullable=False, default="영남대학교 IT관")

class OutdoorPlace(Base):
    __tablename__ = "outdoor_place"

    place_id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    category = Column(String(100))
    address = Column(String(255))
    lon = Column(Double, nullable=False)
    lat = Column(Double, nullable=False)
    has_facility = Column(Boolean, default=False)
    facility_name = Column(String(150))
    facility_type = Column(String(100))
    facility_location = Column(String(255))
    is_accessible = Column(Boolean, default=False)

