import math


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    R = 6371000  # Earth radius in metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def format_distance(metres: int) -> str:
    if metres >= 1000:
        return f'{metres / 1000:.2f} km'
    return f'{metres} m'
