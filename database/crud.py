from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database.models import User, Incident
import datetime


async def get_user_by_username(db: AsyncSession, username: str):
    result = await db.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, username: str,
                      password_hash: str, role: str, room_id: str):
    user = User(
        username=username,
        password_hash=password_hash,
        role=role,
        room_id=room_id,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def update_user_gps(db: AsyncSession, username: str,
                          lat: float, lon: float):
    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()
    if user:
        user.lat = lat
        user.lon = lon
        await db.commit()


async def save_incident(db: AsyncSession, room_id: str,
                        threat_type: str, threat_level: str,
                        confidence: float, cctv_lat: float,
                        cctv_lon: float, distance_m: int,
                        report_text: str):
    incident = Incident(
        room_id=room_id,
        threat_type=threat_type,
        threat_level=threat_level,
        confidence=confidence,
        cctv_lat=cctv_lat,
        cctv_lon=cctv_lon,
        distance_m=distance_m,
        report_text=report_text,
        timestamp=datetime.datetime.utcnow(),
        status='active',
    )
    db.add(incident)
    await db.commit()
    await db.refresh(incident)
    return incident


async def get_incidents(db: AsyncSession, room_id: str = None):
    if room_id:
        result = await db.execute(
            select(Incident)
            .where(Incident.room_id == room_id)
            .order_by(Incident.timestamp.desc())
            .limit(50)
        )
    else:
        result = await db.execute(
            select(Incident)
            .order_by(Incident.timestamp.desc())
            .limit(50)
        )
    return result.scalars().all()
