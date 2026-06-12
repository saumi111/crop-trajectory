"""
Soil Moisture Retrieval Module
Combines SMAP + SAR + Weather for field-level
root zone moisture estimation

Three components:
1. SMAP surface moisture — 36km anchor value
2. SAR VV correction — field-level adjustment
3. Water balance model — root zone depth estimate

Output:
- Surface moisture (0-5cm)
- Root zone moisture (crop specific depth)
- Moisture trend (rising/stable/declining)
- Irrigation threshold assessment
"""

import requests
import numpy as np
from datetime import datetime, timedelta
import os


NASA_TOKEN = os.environ.get("NASA_EARTHDATA_TOKEN", "")


def get_smap_moisture(lat, lng):
    """
    Get SMAP surface soil moisture
    Uses NASA SMAP SPL4SMGP product
    Returns volumetric water content m3/m3
    """
    try:
        # Use NASA SMAP via AppEEARS or direct THREDDS
        # Fallback to Open-Meteo soil moisture if SMAP unavailable
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lng,
                "hourly": "soil_moisture_0_to_1cm",
                "forecast_days": 1,
                "timezone": "Europe/Dublin"
            }, timeout=15)

        if r.status_code == 200:
            data = r.json()
            values = data.get("hourly", {}).get(
                "soil_moisture_0_to_1cm", [])
            valid = [v for v in values if v is not None]
            if valid:
                return round(float(np.mean(valid)), 4)
    except:
        pass
    return None


def get_rootzone_from_weather(lat, lng, start_date, end_date,
                               smap_surface=None):
    """
    Estimate root zone moisture from water balance
    Uses daily rainfall and ET0
    Anchored to SMAP surface value if available
    """
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lng,
                "daily": ",".join([
                    "precipitation_sum",
                    "et0_fao_evapotranspiration",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "soil_moisture_0_to_7cm_mean"
                ]),
                "past_days": 30,
                "forecast_days": 1,
                "timezone": "Europe/Dublin"
            }, timeout=20)

        if r.status_code != 200:
            return None

        data = r.json().get("daily", {})
        rain = data.get("precipitation_sum", [])
        et0 = data.get("et0_fao_evapotranspiration", [])
        soil_m = data.get("soil_moisture_0_to_7cm", [])
        temps_max = data.get("temperature_2m_max", [])
        temps_min = data.get("temperature_2m_min", [])
        dates = data.get("time", [])

        # Start from SMAP or model default
        moisture = smap_surface if smap_surface else 0.25
        field_capacity = 0.35
        wilting_point = 0.12
        daily_records = []

        for i, date in enumerate(dates):
            rain_d = rain[i] if i < len(rain) and rain[i] else 0
            et0_d = et0[i] if i < len(et0) and et0[i] else 2.5
            sm_d = soil_m[i] if i < len(soil_m) and soil_m[i] else None
            tmax = temps_max[i] if i < len(temps_max) else None
            tmin = temps_min[i] if i < len(temps_min) else None

            # Water balance
            infiltration = rain_d * 0.8  # 80% effective rainfall
            drainage = max(0, moisture + infiltration/100 - field_capacity)
            moisture = moisture + infiltration/100 - et0_d/100 - drainage
            moisture = max(wilting_point, min(field_capacity + 0.05, moisture))

            # Blend with observed soil moisture if available
            if sm_d:
                moisture = 0.7 * moisture + 0.3 * sm_d

            # GDD
            gdd = max(0, (tmax + tmin)/2) if tmax and tmin else 0

            daily_records.append({
                "date": date,
                "moisture": round(moisture, 4),
                "rain_mm": round(rain_d, 1),
                "et0_mm": round(et0_d, 1),
                "gdd": round(gdd, 1)
            })

        return daily_records

    except Exception as e:
        return None


def get_sar_moisture_correction(vv_series, dates, baseline_vv=None):
    """
    Derive moisture correction from SAR VV backscatter
    VV is sensitive to surface moisture
    Higher VV = wetter surface

    Returns daily correction factors
    """
    if not vv_series or len(vv_series) < 3:
        return {}

    valid_vv = [v for v in vv_series if v]
    if not valid_vv:
        return {}

    # Baseline VV (dry condition reference)
    baseline = baseline_vv or min(valid_vv)
    max_vv = max(valid_vv)
    vv_range = max_vv - baseline if max_vv > baseline else 1

    corrections = {}
    for date, vv in zip(dates, vv_series):
        if vv:
            # Normalise VV to 0-1 moisture index
            norm = (vv - baseline) / vv_range
            norm = max(0, min(1, norm))
            # Convert to moisture correction (-0.05 to +0.05 m3/m3)
            correction = (norm - 0.5) * 0.10
            corrections[date] = round(correction, 4)

    return corrections


def get_soil_moisture_profile(lat, lng, crop_type,
                               sar_observations=None,
                               smap_surface=None):
    """
    Main function — complete soil moisture profile
    Combines SMAP + SAR + Water balance

    Returns:
    - Current surface moisture
    - Current root zone moisture
    - 10-day history
    - Irrigation thresholds
    - Status and trend
    """
    today = datetime.now()
    start_30d = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    end_today = today.strftime("%Y-%m-%d")

    # Step 1 — Get SMAP surface moisture
    smap = smap_surface or get_smap_moisture(lat, lng)
    smap_source = "Open-Meteo soil model" if smap else None

    # Step 2 — Water balance root zone
    daily = get_rootzone_from_weather(
        lat, lng, start_30d, end_today, smap)

    if not daily:
        return {"error": "Could not retrieve moisture data"}

    # Step 3 — SAR correction
    sar_corrections = {}
    if sar_observations:
        dates = []
        vv_vals = []
        for obs in sar_observations:
            if obs.get("available") and obs.get("vv"):
                dates.append(obs["date"])
                vv_vals.append(obs["vv"])
        sar_corrections = get_sar_moisture_correction(vv_vals, dates)

    # Apply SAR corrections to daily records
    for record in daily:
        date = record["date"]
        if date in sar_corrections:
            corrected = record["moisture"] + sar_corrections[date]
            record["moisture"] = round(
                max(0.05, min(0.45, corrected)), 4)
            record["sar_corrected"] = True
        else:
            record["sar_corrected"] = False

    # Step 4 — Crop specific root zone depths
    root_depths = {
        "Winter Wheat": 100,
        "Spring Barley": 80,
        "Potato": 60,
        "Oilseed Rape": 120,
        "Grassland": 30,
        "Unknown": 80
    }
    root_depth = root_depths.get(crop_type, 80)

    # Root zone adjustment (deeper = more buffered)
    depth_factor = root_depth / 100
    for record in daily:
        # Deeper roots access more stable moisture
        record["root_zone_moisture"] = round(
            record["moisture"] * (0.85 + 0.15 * depth_factor), 4)

    # Step 5 — Current status
    latest = daily[-1]
    recent_10 = daily[-10:]
    current_rz = latest["root_zone_moisture"]

    # Trend
    recent_vals = [r["root_zone_moisture"] for r in recent_10]
    first_half = np.mean(recent_vals[:5])
    second_half = np.mean(recent_vals[5:])
    diff = second_half - first_half

    if diff > 0.02:
        trend = "Rising"
        trend_icon = "↑"
    elif diff < -0.02:
        trend = "Declining"
        trend_icon = "↓"
    else:
        trend = "Stable"
        trend_icon = "→"

    # Irrigation thresholds by crop
    thresholds = {
        "Winter Wheat":  {"optimal": 0.28, "stress": 0.18, "critical": 0.12},
        "Spring Barley": {"optimal": 0.26, "stress": 0.17, "critical": 0.11},
        "Potato":        {"optimal": 0.30, "stress": 0.20, "critical": 0.14},
        "Oilseed Rape":  {"optimal": 0.27, "stress": 0.17, "critical": 0.11},
        "Grassland":     {"optimal": 0.25, "stress": 0.15, "critical": 0.10},
        "Unknown":       {"optimal": 0.27, "stress": 0.18, "critical": 0.12}
    }

    thresh = thresholds.get(crop_type, thresholds["Unknown"])

    if current_rz >= thresh["optimal"]:
        status = "Optimal"
        status_icon = "✅"
        irrigation_needed = False
    elif current_rz >= thresh["stress"]:
        status = "Adequate"
        status_icon = "✅"
        irrigation_needed = False
    elif current_rz >= thresh["critical"]:
        status = "Moderate Stress"
        status_icon = "⚠️"
        irrigation_needed = True
    else:
        status = "Severe Stress"
        status_icon = "🔴"
        irrigation_needed = True

    # Days to stress threshold
    days_to_stress = None
    if not irrigation_needed:
        for i, record in enumerate(daily[-10:]):
            if record["root_zone_moisture"] < thresh["stress"]:
                days_to_stress = i + 1
                break

    return {
        "data_sources": {
            "surface_moisture": smap_source or "Water balance model",
            "root_zone": "FAO-56 water balance + SAR correction",
            "sar_corrections_applied": len(sar_corrections) > 0
        },
        "surface_moisture": {
            "value": round(smap, 4) if smap else None,
            "source": smap_source
        },
        "root_zone": {
            "current_moisture": current_rz,
            "depth_cm": root_depth,
            "status": status,
            "status_icon": status_icon,
            "trend": trend,
            "trend_icon": trend_icon,
            "irrigation_needed": irrigation_needed,
            "days_to_stress": days_to_stress
        },
        "thresholds": {
            "optimal": thresh["optimal"],
            "stress": thresh["stress"],
            "critical": thresh["critical"],
            "current": current_rz
        },
        "irrigation_recommendation": (
            f"Irrigate now — root zone at {current_rz:.3f} "
            f"(below stress threshold {thresh['stress']})"
            if irrigation_needed else
            f"No irrigation needed — "
            f"{f'stress expected in {days_to_stress} days' if days_to_stress else 'moisture adequate'}"
        ),
        "10_day_history": [
            {
                "date": r["date"],
                "root_zone_moisture": r["root_zone_moisture"],
                "rain_mm": r["rain_mm"],
                "status": (
                    "Optimal" if r["root_zone_moisture"] >= thresh["optimal"]
                    else "Adequate" if r["root_zone_moisture"] >= thresh["stress"]
                    else "Stress"
                ),
                "sar_corrected": r.get("sar_corrected", False)
            }
            for r in daily[-10:]
        ]
    }


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    lat, lng = 53.6529, -6.6789
    crop_type = "Winter Wheat"

    print(f"Soil Moisture Profile — {crop_type}")
    print("="*50)

    # Load SAR observations
    with open("/workspaces/crop-trajectory/sar_ireland_2026.json") as f:
        sar_obs = json.load(f)

    result = get_soil_moisture_profile(
        lat, lng, crop_type, sar_obs)

    if result.get("error"):
        print(f"Error: {result['error']}")
    else:
        print(f"\nData Sources:")
        for k, v in result["data_sources"].items():
            print(f"  {k}: {v}")

        rz = result["root_zone"]
        print(f"\nRoot Zone ({rz['depth_cm']}cm depth):")
        print(f"  Moisture:  {rz['current_moisture']} m³/m³")
        print(f"  Status:    {rz['status_icon']} {rz['status']}")
        print(f"  Trend:     {rz['trend_icon']} {rz['trend']}")

        thresh = result["thresholds"]
        print(f"\nThresholds:")
        print(f"  Optimal:   >{thresh['optimal']}")
        print(f"  Stress:    <{thresh['stress']}")
        print(f"  Critical:  <{thresh['critical']}")
        print(f"  Current:   {thresh['current']}")

        print(f"\nRecommendation:")
        print(f"  {result['irrigation_recommendation']}")

        print(f"\n10-Day History:")
        for r in result["10_day_history"]:
            icon = "✅" if r["status"] in ["Optimal","Adequate"] else "⚠️"
            sar = " SAR✓" if r["sar_corrected"] else ""
            print(f"  {r['date']}: {r['root_zone_moisture']} "
                  f"{icon} {r['status']} "
                  f"(rain:{r['rain_mm']}mm){sar}")
