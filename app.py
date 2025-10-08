from fastapi import FastAPI
import logging
from database import engine, Base
# 새로 만든 prediction_router를 임포트 목록에 추가합니다.
from routers import favorites_router, locations_router, user_router, prediction_router, outdoor_places_router
# services 임포트는 이제 app.py에서 직접 사용하지 않으므로 삭제해도 됩니다.
# import schemas도 직접 사용하지 않으므로 삭제 가능합니다.

logging.basicConfig(level=logging.INFO, format='%(levelname)s:     %(message)s')

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Midas API - Refactored",
    description="전체 프로젝트 구조가 개선된 API 서버입니다.",
    version="10.0.0",
)

# --- 라우터 등록 ---
# 모든 API 엔드포인트들을 이곳에서 깔끔하게 등록하고 관리합니다.
app.include_router(user_router.router, prefix="/users", tags=["users"])
app.include_router(favorites_router.router, prefix="/favorites", tags=["favorites"])
app.include_router(locations_router.router, prefix="/locations", tags=["Predicted Locations"])
# 새로 만든 예측 라우터를 등록합니다. prefix가 없으므로 경로가 바로 /predict가 됩니다.
app.include_router(prediction_router.router, tags=["Prediction"])

# @app.post("/predict") ... 로 시작하던 긴 함수 블록 전체가 삭제됩니다.


app.include_router(
    outdoor_places_router.router, 
    prefix="/outdoor-places", 
    tags=["Outdoor Places"]
)

