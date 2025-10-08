from pydantic import BaseModel, ConfigDict
from typing import Optional

class PredictedLocationBase(BaseModel):
    location_name: Optional[str] = None
    description: Optional[str] = None
    floor: int
    address: str

class PredictedLocationCreate(PredictedLocationBase):
    id: str

class PredictedLocation(PredictedLocationBase):
    id: str
    model_config = ConfigDict(from_attributes=True)

class PredictedLocationUpdate(BaseModel):
    location_name: Optional[str] = None
    description: Optional[str] = None
    floor: Optional[int] = None
    address: Optional[str] = None
