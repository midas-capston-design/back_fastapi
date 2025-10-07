from pydantic import BaseModel, ConfigDict
from typing import Optional

# --- 공통 필드를 위한 Base 모델 ---
class OutdoorPlaceBase(BaseModel):
    name: str
    category: Optional[str] = None
    address: Optional[str] = None
    lon: float
    lat: float
    has_facility: Optional[bool] = False
    facility_name: Optional[str] = None
    facility_type: Optional[str] = None
    facility_location: Optional[str] = None
    is_accessible: Optional[bool] = False

# --- 데이터 생성을 위한 Create 모델 ---
class OutdoorPlaceCreate(OutdoorPlaceBase):
    place_id: str

# --- 데이터 수정을 위한 Update 모델 (모든 필드 선택적) ---
class OutdoorPlaceUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    address: Optional[str] = None
    lon: Optional[float] = None
    lat: Optional[float] = None
    has_facility: Optional[bool] = None
    facility_name: Optional[str] = None
    facility_type: Optional[str] = None
    facility_location: Optional[str] = None
    is_accessible: Optional[bool] = None

# --- API 응답을 위한 Read 모델 ---
class OutdoorPlace(OutdoorPlaceBase):
    place_id: str
    model_config = ConfigDict(from_attributes=True)