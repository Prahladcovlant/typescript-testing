import math


def haversine(lat1: float, lon1: float, lat2: float, lon2: float):
    radius_earth_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = radius_earth_km * c
    distance_miles = distance_km * 0.621371
    return {
        "distance_km": round(distance_km, 4),
        "distance_miles": round(distance_miles, 4),
    }

