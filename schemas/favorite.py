from pydantic import BaseModel, ConfigDict
from typing import Optional
from enum import Enum

class PlaceCategory(str, Enum):
    home = "home"; work = "work"; convenienceStore = "convenienceStore"
    school = "school"; etc = "etc"

class FavoriteType(str, Enum):
    place = "place"; bus = "bus"; busStop = "busStop"

class FavoriteCreate(BaseModel):
    id: str
    type: FavoriteType
    name: str
    address: Optional[str] = None
    place_category: Optional[PlaceCategory] = None
    bus_number: Optional[str] = None
    station_name: Optional[str] = None
    station_id: Optional[str] = None

class FavoriteResponse(FavoriteCreate):
    model_config = ConfigDict(from_attributes=True)
