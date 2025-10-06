from pydantic import BaseModel, ConfigDict, EmailStr
from typing import Optional

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

# --- 추가된 부분 ---
# 사용자 이름 변경을 위한 요청 스키마
class UserNameUpdate(BaseModel):
    userName: str