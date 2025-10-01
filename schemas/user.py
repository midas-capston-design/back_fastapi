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
