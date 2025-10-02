from pydantic import BaseModel
from typing import Optional, List

class SensorInput(BaseModel):
    Mag_X: float; Mag_Y: float; Mag_Z: float
    Ori_X: float; Ori_Y: float; Ori_Z: float

class ModelOutput(BaseModel):
    prediction: int
    confidence: Optional[float] = None
    top_k_results: List[str]
