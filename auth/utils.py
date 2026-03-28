from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi import HTTPException, status


SECRET_KEY         = 'sentinel_secret_key_change_before_production'
ALGORITHM          = 'HS256'
TOKEN_EXPIRE_HOURS = 24

pwd_ctx = CryptContext(schemes=['bcrypt'], deprecated='auto')


def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_token(data: dict) -> str:
    payload = data.copy()
    payload['exp'] = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid or expired token'
        )
