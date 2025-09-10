from pydantic import BaseModel, field_validator, EmailStr, ConfigDict
from typing import Optional, List
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
    model_config = ConfigDict(from_attributes=True)

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

    @field_validator('address', 'place_category', mode='before')
    def check_place_fields(cls, v, values):
        if values.data.get('type') == 'place' and v is None:
            raise ValueError('type이 place일 경우 address와 place_category는 필수입니다.')
        return v
    @field_validator('bus_number', mode='before')
    def check_bus_fields(cls, v, values):
        if values.data.get('type') == 'bus' and v is None:
            raise ValueError('type이 bus일 경우 bus_number는 필수입니다.')
        return v
    @field_validator('station_name', 'station_id', mode='before')
    def check_bus_stop_fields(cls, v, values):
        if values.data.get('type') == 'busStop' and v is None:
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
    model_config = ConfigDict(from_attributes=True)

# --- 예측 위치(Predicted Location) 관련 스키마 ---
class PredictedLocationBase(BaseModel):
    # 🔽 [수정] location_name을 Optional[str]로 변경하여 None 값을 허용
    location_name: Optional[str] = None
    description: Optional[str] = None
    floor: int
    address: str

class PredictedLocationCreate(PredictedLocationBase):
    id: int

class PredictedLocation(PredictedLocationBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class PredictedLocationUpdate(BaseModel):
    location_name: Optional[str] = None
    description: Optional[str] = None
    floor: Optional[int] = None
    address: Optional[str] = None

# --- 모델 예측 입출력 스키마 ---
class SensorInput(BaseModel):
    Mag_X: float
    Mag_Y: float
    Mag_Z: float
    Ori_X: float
    Ori_Y: float
    Ori_Z: float
    top_k: Optional[int] = 1

class ModelOutput(BaseModel):
    prediction: int
    confidence: Optional[float] = None
    location_details: Optional[PredictedLocation] = None
    top_k_results: Optional[List[str]] = None

