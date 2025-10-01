# -*- coding: utf-8 -*-
"""
보안 관련 유틸리티 모듈

이 파일은 비밀번호 암호화와 같이 보안과 직접적으로 관련된 함수들을 모아놓은 곳입니다.
인증(auth)이나 데이터베이스 처리(crud) 등 다른 모듈에서 이 함수들을 가져와 사용합니다.
보안 관련 코드를 별도 파일로 분리하면, 순환 참조(circular import) 문제를 방지하고
코드의 역할을 명확하게 분리할 수 있습니다.
"""

from passlib.context import CryptContext

# --- 비밀번호 암호화 컨텍스트 설정 ---
# CryptContext: 비밀번호 해싱(hashing)을 위한 라이브러리입니다.
# schemes=["bcrypt"]: 사용할 해싱 알고리즘을 bcrypt로 지정합니다. bcrypt는 현재
#                    비밀번호 해싱에 가장 널리 추천되는 안전한 알고리즘 중 하나입니다.
# deprecated="auto": 구 버전의 해시 형식이 발견되면, 새로운 형식으로 자동 업데이트하도록 설정합니다.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    입력된 평문 비밀번호와 데이터베이스에 저장된 해시된 비밀번호를 비교합니다.

    Args:
        plain_password (str): 사용자가 로그인 시 입력한 비밀번호 원문.
        hashed_password (str): 데이터베이스에 저장되어 있는, 해싱된 비밀번호.

    Returns:
        bool: 비밀번호가 일치하면 True, 그렇지 않으면 False를 반환합니다.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    평문 비밀번호를 받아 bcrypt 알고리즘으로 해싱하여 반환합니다.

    Args:
        password (str): 암호화할 비밀번호 원문.

    Returns:
        str: 해싱된 비밀번호 문자열.
    """
    return pwd_context.hash(password)
