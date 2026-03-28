from sqlalchemy import Column, String, Float, DateTime, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import datetime


DATABASE_URL = 'sqlite+aiosqlite:///./sentinel.db'

engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String,  unique=True, index=True, nullable=False)
    password_hash = Column(String,  nullable=False)
    role          = Column(String,  nullable=False)   # 'cctv' or 'police'
    room_id       = Column(String,  nullable=True)
    lat           = Column(Float,   nullable=True)
    lon           = Column(Float,   nullable=True)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)


class Incident(Base):
    __tablename__ = 'incidents'

    id           = Column(Integer, primary_key=True, index=True)
    room_id      = Column(String,  index=True, nullable=False)
    threat_type  = Column(String,  nullable=False)
    threat_level = Column(String,  nullable=False)
    confidence   = Column(Float,   nullable=True)
    cctv_lat     = Column(Float,   nullable=True)
    cctv_lon     = Column(Float,   nullable=True)
    distance_m   = Column(Integer, nullable=True)
    report_text  = Column(Text,    nullable=True)
    timestamp    = Column(DateTime, default=datetime.datetime.utcnow)
    status       = Column(String,  default='active')


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
