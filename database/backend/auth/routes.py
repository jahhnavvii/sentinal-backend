from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from database.models import get_db
from database.crud import get_user_by_username, create_user
from auth.utils import hash_password, verify_password, create_token
import random, string

router = APIRouter()


def generate_room_id(length: int = 6) -> str:
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str          # 'cctv' or 'police'
    room_id: str = None   # police fills this from QR scan


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token:    str
    username: str
    role:     str
    room_id:  str


@router.post('/register', response_model=AuthResponse)
async def register(req: RegisterRequest,
                   db: AsyncSession = Depends(get_db)):
    existing = await get_user_by_username(db, req.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Username already taken'
        )
    if req.role not in ('cctv', 'police'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Role must be cctv or police'
        )
        
    # In Hackathon flow: both roles get auto-generated room if none provided
    room_id = req.room_id if req.room_id else generate_room_id()
    password_hash = hash_password(req.password)
    user = await create_user(db, req.username,
                             password_hash, req.role, room_id)
    token = create_token({
        'sub':     user.username,
        'role':    user.role,
        'room_id': user.room_id,
    })
    return AuthResponse(token=token, username=user.username,
                        role=user.role, room_id=user.room_id)


@router.post('/login', response_model=AuthResponse)
async def login(req: LoginRequest,
                db: AsyncSession = Depends(get_db)):
    user = await get_user_by_username(db, req.username)
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid username or password'
        )
    token = create_token({
        'sub':     user.username,
        'role':    user.role,
        'room_id': user.room_id,
    })
    return AuthResponse(token=token, username=user.username,
                        role=user.role, room_id=user.room_id)
