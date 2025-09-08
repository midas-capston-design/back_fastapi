# schemas.py

from pydantic import BaseModel, validator, EmailStr
from typing import Optional
from enum import Enum

# --- 사용자(User) 관련 스키마 ---

class UserBase(BaseModel):
    userId: str
    userName: str
    email: EmailStr
    phone_number: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    userId: Optional[str] = None


# --- 즐겨찾기(Favorite) 관련 스키마 ---

class PlaceCategory(str, Enum):
    home = "home"
    work = "work"
    convenienceStore = "convenienceStore"
    school = "school"
    etc = "etc"

class FavoriteType(str, Enum):
    place = "place"
    bus = "bus"
    busStop = "busStop"

class FavoriteCreate(BaseModel):
    id: str
    type: FavoriteType
    name: str
    address: Optional[str] = None
    place_category: Optional[PlaceCategory] = None
    bus_number: Optional[str] = None
    station_name: Optional[str] = None
    station_id: Optional[str] = None

    @validator('address', 'place_category', always=True)
    def check_place_fields(cls, v, values):
        if values.get('type') == 'place' and v is None:
            raise ValueError('type이 place일 경우 address와 place_category는 필수입니다.')
        return v

    @validator('bus_number', always=True)
    def check_bus_fields(cls, v, values):
        if values.get('type') == 'bus' and v is None:
            raise ValueError('type이 bus일 경우 bus_number는 필수입니다.')
        return v
    
    @validator('station_name', 'station_id', always=True)
    def check_bus_stop_fields(cls, v, values):
        if values.get('type') == 'busStop' and v is None:
            raise ValueError('type이 busStop일 경우 station_name과 station_id는 필수입니다.')
        return v

class FavoriteResponse(BaseModel):
    id: str
    type: FavoriteType
    name: str
    address: Optional[str] = None
    place_category: Optional[PlaceCategory] = None
    bus_number: Optional[str] = None
    station_name: Optional[str] = None
    station_id: Optional[str] = None

    class Config:
        orm_mode = True



# --- 예측 위치(Predicted Location) 관련 스키마 ---

class PredictedLocationBase(BaseModel):
    location_name: str
    description: Optional[str] = None

class PredictedLocationCreate(PredictedLocationBase):
    id: int # 생성 시에는 id도 함께 받습니다.

class PredictedLocation(PredictedLocationBase):
    id: int

    class Config:
        orm_mode = True

class PredictedLocationUpdate(BaseModel):
    location_name: Optional[str] = None
    description: Optional[str] = None

