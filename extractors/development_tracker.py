"""
Crop Development Tracker
Combines SAR + NDVI + Weather into development velocity
Compares current season against historical baseline
"""
import json
import datetime
import requests
import numpy as np
import os

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

def get_token():
    r = requests.post(CDSE_TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": os.environ.get("CDSE_CLIENT_ID"),
        "client_secret": os.environ.get("CDSE_CLIENT_SECRET")
    }, timeout=10)
    r.raise_for_status()
    return r.json().get("access_token")

def get_ndvi(lat, lng, date, token):
    """Get Sentinel-2 NDVI for a date"""
    d = datetime.datetime.strptime(date, "%Y-%m-%d")
    d_from = (d - datetime.timedelta(days=10)).strftime("%Y-%m-%dT00:00:00Z")
    d_to = (d + datetime.timedelta(days=3)).strftime("%Y-%m-%dT23:59:59Z")
    delta = 0.02
    bbox = [lng-delta, lat-delta, lng+delta, lat+delta]

    evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{bands:["B04","B08","B05"],units:"REFLECTANCE"}],
    output: {bands:2}
  };
}
function evaluatePixel(s) {
  var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 0.0001);
  var ndre = (s.B08 - s.B05) / (s.B08 + s.B05 + 0.0001);
  return [ndvi, ndre];
}
"""
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": d_from, "to": d_to},
                    "maxCloudCoverage": 50
                }
            }]
        },
        "evalscript": evalscript,
        "output": {
            "width": 5, "height": 5,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        }
    }
    try:
        r = requests.post(CDSE_PROCESS_URL, json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=30)
        if r.status_code == 200:
            import tifffile, io
            arr = tifffile.imread(io.BytesIO(r.content))
            if arr is not None and arr.size > 0:
                if len(arr.shape) == 3:
                    ndvi_vals = arr[:,:,0].flatten()
                    ndre_vals = arr[:,:,1].flatten()
                else:
                    ndvi_vals = arr.flatten()
                    ndre_vals = arr.flatten()
                ndvi_valid = ndvi_vals[(ndvi_vals > 0.05) & (ndvi_vals < 1.0)]
                ndre_valid = ndre_vals[(ndre_vals > -0.5) & (ndre_vals < 1.0)]
                if len(ndvi_valid) > 0:
                    return {
                        "ndvi": round(float(np.mean(ndvi_valid)), 4),
                        "ndre": round(float(np.mean(ndre_valid)), 4) if len(ndre_valid) > 0 else None
                    }
        return {"ndvi": None, "ndre": None}
    except Exception as e:
        return {"ndvi": None, "ndre": None, "error": str(e)}

def get_weather(lat, lng, date):
    """Get weather for a specific date"""
    d = datetime.datetime.strptime(date, "%Y-%m-%d")
    d_from = (d - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    d_to = (d + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        for attempt in range(3):
            try:
                r = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": lat, "longitude": lng,
                        "start_date": date, "end_date": date,
                        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                        "timezone": "Europe/London"
                    }, timeout=20)
                break
            except requests.exceptions.Timeout:
                if attempt == 2:
                    raise
                import time; time.sleep(2)
        if r.status_code == 200:
            data = r.json().get("daily", {})
            temps_max = data.get("temperature_2m_max", [])
            temps_min = data.get("temperature_2m_min", [])
            rain = data.get("precipitation_sum", [])
            temp_max = temps_max[0] if len(temps_max) > 0 else None
            temp_min = temps_min[0] if len(temps_min) > 0 else None
            rain_val = rain[0] if len(rain) > 0 else None
            temp_avg = round((temp_max + temp_min) / 2, 1) if temp_max and temp_min else None
            gdd = round(max(0, temp_avg - 0), 1) if temp_avg else None  # Base 0°C for wheat
            return {
                "temp_max": temp_max,
                "temp_min": temp_min,
                "temp_avg": temp_avg,
                "rain_mm": rain_val,
                "gdd": gdd  # Growing degree days
            }
    except Exception as e:
        print(f"Weather error for {date}: {e}")
    return {"temp_max": None, "temp_min": None, "temp_avg": None, "rain_mm": None, "gdd": None}

def calculate_velocity(observations):
    """
    Calculate development velocity between observations
    velocity = (current_value - previous_value) / days_between
    """
    for i in range(1, len(observations)):
        curr = observations[i]
        prev = observations[i-1]
        
        # Days between observations
        d1 = datetime.datetime.strptime(prev["date"], "%Y-%m-%d")
        d2 = datetime.datetime.strptime(curr["date"], "%Y-%m-%d")
        days = (d2 - d1).days
        
        if days > 0:
            # RVI velocity
            if curr.get("rvi") and prev.get("rvi"):
                curr["rvi_velocity"] = round((curr["rvi"] - prev["rvi"]) / days, 5)
            else:
                curr["rvi_velocity"] = None
                
            # NDVI velocity
            if curr.get("ndvi") and prev.get("ndvi"):
                curr["ndvi_velocity"] = round((curr["ndvi"] - prev["ndvi"]) / days, 5)
            else:
                curr["ndvi_velocity"] = None
        else:
            curr["rvi_velocity"] = None
            curr["ndvi_velocity"] = None
    
    observations[0]["rvi_velocity"] = None
    observations[0]["ndvi_velocity"] = None
    return observations

def build_season_profile(lat, lng, start_date, end_date, interval_days=12):
    """
    Build complete season profile combining SAR + NDVI + Weather
    """
    print(f"\nBuilding season profile: {start_date} to {end_date}")
    token = get_token()
    
    observations = []
    current = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"  Processing {date_str}...")
        
        obs = {"date": date_str}
        
        # SAR
        from extractors.sar_timeseries import get_sar_value
        sar = get_sar_value(lat, lng, date_str, token)
        obs["vv"] = sar.get("vv")
        obs["vh"] = sar.get("vh")
        obs["rvi"] = sar.get("rvi")
        obs["sar_available"] = sar.get("available", False)
        
        # NDVI
        ndvi_data = get_ndvi(lat, lng, date_str, token)
        obs["ndvi"] = ndvi_data.get("ndvi")
        obs["ndre"] = ndvi_data.get("ndre")
        
        # Weather
        wx = get_weather(lat, lng, date_str)
        obs["temp_avg"] = wx.get("temp_avg")
        obs["rain_mm"] = wx.get("rain_mm")
        obs["gdd"] = wx.get("gdd")
        
        observations.append(obs)
        current += datetime.timedelta(days=interval_days)
    
    # Calculate velocities
    observations = calculate_velocity(observations)
    
    return observations

def compare_seasons(season_a, season_b):
    """
    Compare two seasons by day-of-year
    Returns deviation of season_b from season_a
    """
    # Index by day of year
    def doy(date_str):
        d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return d.timetuple().tm_yday

    a_index = {doy(o["date"]): o for o in season_a}
    b_index = {doy(o["date"]): o for o in season_b}
    
    deviations = []
    for day in sorted(b_index.keys()):
        if day in a_index:
            a_rvi = a_index[day].get("rvi")
            b_rvi = b_index[day].get("rvi")
            a_ndvi = a_index[day].get("ndvi")
            b_ndvi = b_index[day].get("ndvi")
            
            dev = {"day_of_year": day}
            
            if a_rvi and b_rvi:
                dev["rvi_deviation"] = round(b_rvi - a_rvi, 4)
                dev["rvi_deviation_pct"] = round((b_rvi - a_rvi) / a_rvi * 100, 1)
            
            if a_ndvi and b_ndvi:
                dev["ndvi_deviation"] = round(b_ndvi - a_ndvi, 4)
                dev["ndvi_deviation_pct"] = round((b_ndvi - a_ndvi) / a_ndvi * 100, 1)
            
            # Flag significant deviation
            rvi_dev = abs(dev.get("rvi_deviation_pct", 0))
            ndvi_dev = abs(dev.get("ndvi_deviation_pct", 0))
            dev["flag"] = rvi_dev > 15 or ndvi_dev > 20
            dev["severity"] = "HIGH" if rvi_dev > 25 or ndvi_dev > 30 else "MEDIUM" if dev["flag"] else "OK"
            
            deviations.append(dev)
    
    return deviations

if __name__ == "__main__":
    lat, lng = 53.2307, -0.5406
    
    # Load existing SAR data
    with open("sar_timeseries_test.json") as f:
        sar_2025 = json.load(f)
    
    print("=== Adding NDVI and Weather to SAR time series ===")
    
    # Enrich first 5 observations with NDVI + weather as test
    token = get_token()
    enriched = []
    
    for obs in sar_2025[:6]:
        date = obs["date"]
        print(f"Enriching {date}...")
        
        ndvi_data = get_ndvi(lat, lng, date, token)
        wx = get_weather(lat, lng, date)
        
        enriched_obs = {**obs,
            "ndvi": ndvi_data.get("ndvi"),
            "ndre": ndvi_data.get("ndre"),
            "temp_avg": wx.get("temp_avg"),
            "rain_mm": wx.get("rain_mm"),
            "gdd": wx.get("gdd")
        }
        enriched.append(enriched_obs)
        print(f"  SAR RVI: {obs.get('rvi')} | NDVI: {ndvi_data.get('ndvi')} | Temp: {wx.get('temp_avg')}°C | Rain: {wx.get('rain_mm')}mm")
    
    # Calculate velocity
    enriched = calculate_velocity(enriched)
    
    print("\n=== Development Velocity ===")
    for o in enriched:
        print(f"{o['date']}: RVI={o.get('rvi')} vel={o.get('rvi_velocity')} | NDVI={o.get('ndvi')} vel={o.get('ndvi_velocity')}")
    
    with open("development_tracker_test.json", "w") as f:
        json.dump(enriched, f, indent=2)
    print("\nSaved to development_tracker_test.json")
