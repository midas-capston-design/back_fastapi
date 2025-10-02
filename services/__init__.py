# services 폴더의 주요 함수들을 패키지 레벨로 가져옵니다.
# 이렇게 하면 다른 파일에서 from services.model_service import run_prediction 대신
# from services import run_prediction 처럼 간단하게 임포트할 수 있습니다.
# 웨이브렛 코드 적용하면서 추가
from .model_service import run_prediction, MODEL_LOADED, get_model_status
