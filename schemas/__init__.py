# schemas 폴더의 각 파일에 있는 클래스들을 패키지 레벨로 가져옵니다.
from .user import User, UserCreate, Token, TokenData
from .favorite import FavoriteCreate, FavoriteResponse, PlaceCategory, FavoriteType
from .location import PredictedLocation, PredictedLocationCreate, PredictedLocationUpdate
from .prediction import ModelOutput, SensorInput