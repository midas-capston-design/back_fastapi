# -*- coding: utf-8 -*-
"""
Pydantic 스키마 정의 모듈

이 파일은 API의 데이터 유효성 검증을 위한 Pydantic 모델(스키마)을 정의합니다.
- API 요청 및 응답의 데이터 형식을 강제하여 안정성을 높입니다.
- 데이터 타입을 자동으로 변환하고, 유효하지 않은 데이터는 에러를 발생시킵니다.
- FastAPI의 API 문서에 모델 정보가 자동으로 연동되어 명세가 명확해집니다.
"""

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator
from typing import Optional, List
from enum import Enum

# --- 사용자(User) 관련 스키마 ---
# 각 스키마는 특정 API 엔드포인트의 요청 또는 응답 본문(body)에 해당합니다.

class UserBase(BaseModel):
    """사용자 정보의 기본 필드를 정의하는 스키마"""
    userId: str
    userName: str
    email: EmailStr  # Pydantic이 제공하는 이메일 형식 검증 타입
    phone_number: str

class UserCreate(UserBase):
    """회원가입(사용자 생성) 시 요청 본문에 사용되는 스키마. 기본 정보에 비밀번호를 추가로 받음."""
    password: str

class User(UserBase):
    """API 응답으로 반환될 때 사용되는 사용자 정보 스키마. 비밀번호 등 민감 정보는 제외."""
    id: int

    # model_config의 from_attributes=True (구 orm_mode=True) 설정은
    # SQLAlchemy 모델 객체를 Pydantic 스키마로 자동으로 변환할 수 있게 해줍니다.
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    """로그인 성공 시 반환되는 JWT 토큰 정보를 위한 스키마"""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """JWT 토큰을 디코딩했을 때 얻게 되는 payload 데이터의 형식을 정의하는 스키마"""
    userId: Optional[str] = None


# --- 즐겨찾기(Favorite) 관련 스키마 ---

class PlaceCategory(str, Enum):
    """장소 카테고리 Enum"""
    home = "home"; work = "work"; convenienceStore = "convenienceStore"
    school = "school"; etc = "etc"

class FavoriteType(str, Enum):
    """즐겨찾기 타입 Enum"""
    place = "place"; bus = "bus"; busStop = "busStop"

class FavoriteCreate(BaseModel):
    """즐겨찾기 생성을 위한 요청 스키마"""
    id: str
    type: FavoriteType
    name: str
    # 각 필드는 Optional로 선언하여, type에 따라 필요한 필드만 받도록 합니다.
    address: Optional[str] = None
    place_category: Optional[PlaceCategory] = None
    bus_number: Optional[str] = None
    station_name: Optional[str] = None
    station_id: Optional[str] = None

class FavoriteResponse(FavoriteCreate):
    """즐겨찾기 정보를 응답할 때 사용하는 스키마. 현재는 생성 스키마와 동일."""
    # SQLAlchemy 모델 객체로부터 Pydantic 스키마를 생성하기 위해 필요합니다.
    model_config = ConfigDict(from_attributes=True)


# --- 예측 위치(Predicted Location) 관련 스키마 ---
# CRUD(Create, Read, Update, Delete) 각 상황에 맞춰 필요한 필드만 정의하여 사용합니다.

class PredictedLocationBase(BaseModel):
    """예측 위치 정보의 기본 필드를 정의하는 스키마"""
    location_name: Optional[str] = None
    description: Optional[str] = None
    floor: int
    address: str

class PredictedLocationCreate(PredictedLocationBase):
    """예측 위치 정보 '생성' 시 사용되는 스키마. ID를 추가로 받음."""
    id: int

class PredictedLocation(PredictedLocationBase):
    """예측 위치 정보 '조회' 응답 시 사용되는 스키마. ID를 포함."""
    id: int
    model_config = ConfigDict(from_attributes=True)

class PredictedLocationUpdate(BaseModel):
    """예측 위치 정보 '수정' 시 사용되는 스키마. 모든 필드는 선택적으로(Optional) 받음."""
    location_name: Optional[str] = None
    description: Optional[str] = None
    floor: Optional[int] = None
    address: Optional[str] = None


# --- 모델 예측 입출력 스키마 ---

class SensorInput(BaseModel):
    """/predict API에 입력되는 센서 데이터의 형식을 정의하는 스키마"""
    Mag_X: float; Mag_Y: float; Mag_Z: float # 자기장 센서 값
    Ori_X: float; Ori_Y: float; Ori_Z: float # 방위각 센서 값
    top_k: Optional[int] = 1 # 상위 K개의 예측 결과를 요청 (기본값 1)

class ModelOutput(BaseModel):
    """/predict API가 반환하는 최종 예측 결과의 형식을 정의하는 스키마"""
    prediction: int # 최종 예측된 위치 ID
    confidence: Optional[float] = None # 예측에 대한 신뢰도 점수 (0.0 ~ 1.0)
    location_details: Optional[PredictedLocation] = None # 예측된 위치의 상세 정보 (DB에서 조회)
    top_k_results: Optional[List[str]] = None # Top-K 예측 결과 목록

