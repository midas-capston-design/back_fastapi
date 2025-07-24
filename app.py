from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import os
from catboost import CatBoostClassifier
import math

app = FastAPI()

# ✅ 모델 경로만 v6로 변경
model_path = os.path.join("model", "model_v6.cbm")
cat_model = CatBoostClassifier()
cat_model.load_model(model_path)

print("✅ 모델 feature 수:", cat_model.feature_names_)

# ✅ 입력 스키마
class ModelInput(BaseModel):
    Mag_X: float = Field(..., description="자기장 X축")
    Mag_Y: float = Field(..., description="자기장 Y축")
    Mag_Z: float = Field(..., description="자기장 Z축")
    Ori_X: float = Field(..., description="Orientation 방향값 (degree)")

# ✅ 출력 스키마
class ModelOutput(BaseModel):
    prediction: int

@app.post("/predict", response_model=ModelOutput)
def predict(data: ModelInput):
    # Ori_X → radian → sin, cos 변환
    ori_rad = math.radians(data.Ori_X)
    ori_sin = math.sin(ori_rad)
    ori_cos = math.cos(ori_rad)

    # 모델 입력
    features = [data.Mag_X, data.Mag_Y, data.Mag_Z, ori_sin, ori_cos]
    input_array = np.array(features).reshape(1, -1)

    # 예측 + 보정 (+1)
    prediction = cat_model.predict(input_array)
    corrected_prediction = int(prediction[0]) + 1

    return {"prediction": corrected_prediction}
