"""
Field Intelligence Module
Adds five high-value features to the CGM:

1. Field Risk Score — single 0-10 number
2. Yield Limiting Factors — what is reducing yield
3. Crop Stage Timeline — past and future dates
4. Disease Risk Engine — weather + stage based
5. Moisture Forecast — root zone stress prediction
"""

import requests
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# FEATURE 1 — FIELD RISK SCORE
# ─────────────────────────────────────────────

def calculate_field_risk_score(ndvi, ndre, smap_moisture,
                                rain_7d, temp_avg, crop_type,
                                current_stage, lai, canopy_cover):
    """
    Calculate field health score 0-10
    10 = perfect conditions
    0 = severe stress

    Sub-scores:
    - Disease risk (weather based)
    - Moisture stress (SMAP + forecast)
    - Nitrogen status (NDRE)
    - Canopy health (NDVI + LAI)
    """
    scores = {}
    details = {}

    # Canopy health score (NDVI)
    if ndvi:
        if ndvi > 0.7:
            scores["canopy"] = 10
            details["canopy"] = "Excellent canopy"
        elif ndvi > 0.5:
            scores["canopy"] = 7
            details["canopy"] = "Good canopy"
        elif ndvi > 0.3:
            scores["canopy"] = 5
            details["canopy"] = "Moderate canopy"
        else:
            scores["canopy"] = 3
            details["canopy"] = "Poor canopy"
    else:
        scores["canopy"] = 5
        details["canopy"] = "No optical data"

    # Nitrogen score (NDRE)
    if ndre:
        if ndre > 0.4:
            scores["nitrogen"] = 10
            details["nitrogen"] = "Low"
        elif ndre > 0.25:
            scores["nitrogen"] = 7
            details["nitrogen"] = "Low"
        elif ndre > 0.15:
            scores["nitrogen"] = 4
            details["nitrogen"] = "Moderate"
        else:
            scores["nitrogen"] = 2
            details["nitrogen"] = "High"
    else:
        scores["nitrogen"] = 5
        details["nitrogen"] = "Unknown"

    # Moisture stress score
    if smap_moisture:
        if 0.20 <= smap_moisture <= 0.35:
            scores["moisture"] = 10
            details["moisture"] = "Low"
        elif smap_moisture < 0.15:
            scores["moisture"] = 3
            details["moisture"] = "High"
        elif smap_moisture > 0.40:
            scores["moisture"] = 6
            details["moisture"] = "Moderate"
        else:
            scores["moisture"] = 7
            details["moisture"] = "Low"
    else:
        scores["moisture"] = 5
        details["moisture"] = "Unknown"

    # Disease risk score
    disease_risk = "Low"
    if rain_7d and temp_avg:
        if rain_7d > 20 and 15 <= temp_avg <= 25:
            disease_risk = "High"
            scores["disease"] = 3
        elif rain_7d > 10 and temp_avg > 12:
            disease_risk = "Moderate"
            scores["disease"] = 6
        else:
            disease_risk = "Low"
            scores["disease"] = 9
    else:
        scores["disease"] = 6

    details["disease"] = disease_risk

    # Overall score — weighted average
    weights = {
        "canopy": 0.30,
        "nitrogen": 0.25,
        "moisture": 0.25,
        "disease": 0.20
    }

    overall = sum(scores[k] * weights[k] for k in scores)
    overall = round(overall, 1)

    # Risk level
    if overall >= 8:
        risk_level = "Low Risk"
    elif overall >= 6:
        risk_level = "Moderate Risk"
    elif overall >= 4:
        risk_level = "Elevated Risk"
    else:
        risk_level = "High Risk"

    return {
        "field_health_score": overall,
        "risk_level": risk_level,
        "sub_scores": {
            "disease_risk": details["disease"],
            "moisture_stress": details["moisture"],
            "nitrogen_deficiency": details["nitrogen"],
            "canopy_health": details["canopy"]
        },
        "yield_risk": (
            "Low" if overall >= 8 else
            "Medium" if overall >= 6 else
            "High"
        )
    }


# ─────────────────────────────────────────────
# FEATURE 2 — YIELD LIMITING FACTORS
# ─────────────────────────────────────────────

def calculate_yield_limiting_factors(crop_type, ndvi, ndre,
                                      smap_moisture, canopy_cover,
                                      rain_7d, base_yield):
    """
    Identify top yield limiting factors
    Each factor shows % yield reduction
    """
    constraints = []
    yield_adjustment = 0

    # Low canopy cover
    if canopy_cover and canopy_cover < 70:
        penalty = round((70 - canopy_cover) / 70 * 15, 1)
        constraints.append({
            "factor": "Low canopy cover",
            "yield_impact_pct": -penalty,
            "severity": "High" if penalty > 10 else "Moderate"
        })
        yield_adjustment -= penalty

    # Nitrogen limitation (NDRE)
    if ndre and ndre < 0.25:
        penalty = round((0.25 - ndre) / 0.25 * 12, 1)
        constraints.append({
            "factor": "Nitrogen limitation",
            "yield_impact_pct": -penalty,
            "severity": "High" if penalty > 8 else "Moderate"
        })
        yield_adjustment -= penalty

    # Moisture deficit
    if smap_moisture and smap_moisture < 0.18:
        penalty = round((0.18 - smap_moisture) / 0.18 * 10, 1)
        constraints.append({
            "factor": "Moisture deficit",
            "yield_impact_pct": -penalty,
            "severity": "High" if penalty > 7 else "Moderate"
        })
        yield_adjustment -= penalty

    # Low NDVI
    if ndvi and ndvi < 0.5:
        penalty = round((0.5 - ndvi) / 0.5 * 10, 1)
        constraints.append({
            "factor": "Reduced canopy greenness",
            "yield_impact_pct": -penalty,
            "severity": "Moderate"
        })
        yield_adjustment -= penalty

    # Excess moisture
    if smap_moisture and smap_moisture > 0.40:
        penalty = 3.0
        constraints.append({
            "factor": "Waterlogging risk",
            "yield_impact_pct": -penalty,
            "severity": "Moderate"
        })
        yield_adjustment -= penalty

    # Sort by impact
    constraints.sort(key=lambda x: x["yield_impact_pct"])

    # Adjusted yield
    if base_yield:
        adjusted = round(base_yield * (1 + yield_adjustment/100), 2)
    else:
        adjusted = None

    return {
        "base_yield_tha": base_yield,
        "adjusted_yield_tha": adjusted,
        "total_yield_impact_pct": round(yield_adjustment, 1),
        "limiting_factors": constraints[:3],  # Top 3
        "no_constraints": len(constraints) == 0
    }


# ─────────────────────────────────────────────
# FEATURE 3 — CROP STAGE TIMELINE
# ─────────────────────────────────────────────

def build_stage_timeline(crop_type, sowing_date_str,
                          current_stage, heading_date_str=None):
    """
    Build complete crop stage timeline
    Shows past dates and predicts future dates
    """
    if not sowing_date_str:
        return None

    sowing = datetime.strptime(sowing_date_str, "%Y-%m-%d")
    today = datetime.now()

    # Stage timelines by crop (days from sowing)
    timelines = {
        "Winter Wheat": [
            ("Sowing",         0,   False),
            ("Emergence",      21,  False),
            ("Tillering",      60,  False),
            ("Stem Extension", 180, False),
            ("Booting",        210, False),
            ("Heading",        240, False),
            ("Grain Fill",     270, False),
            ("Ripening",       290, False),
            ("Harvest",        310, False)
        ],
        "Spring Barley": [
            ("Sowing",         0,   False),
            ("Emergence",      14,  False),
            ("Tillering",      35,  False),
            ("Stem Extension", 65,  False),
            ("Booting",        80,  False),
            ("Heading",        90,  False),
            ("Grain Fill",     105, False),
            ("Ripening",       120, False),
            ("Harvest",        140, False)
        ],
        "Potato": [
            ("Planting",          0,   False),
            ("Emergence",         21,  False),
            ("Canopy Development",45,  False),
            ("Canopy Closure",    65,  False),
            ("Tuber Initiation",  80,  False),
            ("Tuber Bulking",     100, False),
            ("Senescence",        130, False),
            ("Harvest",           155, False)
        ],
        "Oilseed Rape": [
            ("Sowing",         0,   False),
            ("Emergence",      21,  False),
            ("Rosette",        60,  False),
            ("Stem Extension", 180, False),
            ("Flowering",      240, False),
            ("Pod Development",270, False),
            ("Ripening",       300, False),
            ("Harvest",        330, False)
        ]
    }

    stages = timelines.get(crop_type, timelines["Winter Wheat"])
    timeline = []

    for stage_name, days_offset, _ in stages:
        stage_date = sowing + timedelta(days=days_offset)
        days_from_today = (stage_date - today).days

        if days_from_today < -3:
            status = "complete"
        elif abs(days_from_today) <= 3:
            status = "current"
        else:
            status = "upcoming"

        entry = {
            "stage": stage_name,
            "date": stage_date.strftime("%Y-%m-%d"),
            "status": status,
            "days_from_today": days_from_today
        }

        # Add expected window for harvest
        if stage_name == "Harvest" and status == "upcoming":
            entry["window"] = f"{stage_date.strftime('%d %B')} ± 7 days"

        timeline.append(entry)

    # Find next upcoming stage
    next_stage = next(
        (s for s in timeline if s["status"] == "upcoming"), None)

    return {
        "stages": timeline,
        "next_stage": next_stage,
        "harvest_window": next(
            (s.get("window") for s in timeline
             if s["stage"] == "Harvest"), None)
    }


# ─────────────────────────────────────────────
# FEATURE 4 — DISEASE RISK ENGINE
# ─────────────────────────────────────────────

def calculate_disease_risk(crop_type, current_stage,
                             temp_avg, rain_7d, humidity=None,
                             smap_moisture=None):
    """
    Disease risk based on crop stage + weather conditions
    Returns specific disease risks with reasons
    """
    risks = []
    today = datetime.now()

    # Disease rules by crop
    disease_rules = {
        "Winter Wheat": [
            {
                "disease": "Septoria tritici blotch",
                "conditions": lambda t, r, sm: r and r > 15 and t and t > 10,
                "stages": ["Tillering", "Stem Extension", "Booting"],
                "action": "Consider T1 fungicide application"
            },
            {
                "disease": "Yellow rust",
                "conditions": lambda t, r, sm: t and 8 <= t <= 15,
                "stages": ["Stem Extension", "Booting", "Heading"],
                "action": "Scout for yellow stripe symptoms"
            },
            {
                "disease": "Fusarium ear blight",
                "conditions": lambda t, r, sm: r and r > 10 and t and t > 15,
                "stages": ["Heading", "Grain Fill"],
                "action": "T3 ear wash fungicide if risk high"
            }
        ],
        "Spring Barley": [
            {
                "disease": "Rhynchosporium",
                "conditions": lambda t, r, sm: r and r > 10 and t and t < 20,
                "stages": ["Tillering", "Stem Extension"],
                "action": "Consider T1 fungicide"
            },
            {
                "disease": "Ramularia leaf spot",
                "conditions": lambda t, r, sm: t and t > 15,
                "stages": ["Heading", "Grain Fill"],
                "action": "T2 fungicide at flag leaf"
            }
        ],
        "Oilseed Rape": [
            {
                "disease": "Sclerotinia stem rot",
                "conditions": lambda t, r, sm: r and r > 10 and t and 15 <= t <= 25,
                "stages": ["Flowering"],
                "action": "Fungicide at 20-25% flowering"
            },
            {
                "disease": "Light leaf spot",
                "conditions": lambda t, r, sm: r and r > 8 and t and t < 15,
                "stages": ["Rosette", "Stem Extension"],
                "action": "Autumn/early spring fungicide"
            }
        ],
        "Potato": [
            {
                "disease": "Late blight (P. infestans)",
                "conditions": lambda t, r, sm: r and r > 10 and t and 10 <= t <= 25,
                "stages": ["Canopy Closure", "Tuber Initiation",
                           "Tuber Bulking"],
                "action": "Maintain blight spray programme"
            },
            {
                "disease": "Common scab",
                "conditions": lambda t, r, sm: sm and sm < 0.18,
                "stages": ["Tuber Initiation"],
                "action": "Maintain soil moisture at tuber initiation"
            }
        ]
    }

    crop_rules = disease_rules.get(crop_type, [])

    for rule in crop_rules:
        # Check if current stage matches
        stage_match = any(
            s.lower() in (current_stage or "").lower()
            for s in rule["stages"]
        )

        # Check weather conditions
        weather_match = rule["conditions"](
            temp_avg, rain_7d, smap_moisture)

        if stage_match and weather_match:
            risk_level = "High"
        elif weather_match or stage_match:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        reasons = []
        if stage_match:
            reasons.append(f"Susceptible growth stage: {current_stage}")
        if rain_7d and rain_7d > 10:
            reasons.append(f"Recent rainfall: {rain_7d}mm in 7 days")
        if temp_avg and 10 <= temp_avg <= 25:
            reasons.append(f"Favourable temperature: {temp_avg}°C")

        risks.append({
            "disease": rule["disease"],
            "risk_level": risk_level,
            "reasons": reasons,
            "recommended_action": rule["action"]
        })

    # Sort by risk level
    risk_order = {"High": 0, "Moderate": 1, "Low": 2}
    risks.sort(key=lambda x: risk_order.get(x["risk_level"], 3))

    return {
        "disease_risks": risks,
        "highest_risk": risks[0]["risk_level"] if risks else "Low",
        "priority_action": risks[0]["recommended_action"] if risks else None
    }


# ─────────────────────────────────────────────
# FEATURE 5 — MOISTURE FORECAST
# ─────────────────────────────────────────────

def get_moisture_forecast(lat, lng, crop_type, smap_current=None):
    """
    Root zone moisture forecast for next 10 days
    Uses Open-Meteo forecast API
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
                    "temperature_2m_min"
                ]),
                "forecast_days": 10,
                "timezone": "Europe/Dublin"
            }, timeout=15)

        if r.status_code != 200:
            return None

        data = r.json().get("daily", {})
        dates = data.get("time", [])
        rain = data.get("precipitation_sum", [])
        et0 = data.get("et0_fao_evapotranspiration", [])
        temps_max = data.get("temperature_2m_max", [])
        temps_min = data.get("temperature_2m_min", [])

        # Simulate root zone moisture
        # Start from current SMAP value or default
        moisture = smap_current or 0.25
        forecast = []

        for i, date in enumerate(dates):
            rain_d = rain[i] if i < len(rain) and rain[i] else 0
            et0_d = et0[i] if i < len(et0) and et0[i] else 3.0

            # Simple water balance
            moisture += (rain_d * 0.01) - (et0_d * 0.008)
            moisture = max(0.05, min(0.45, moisture))

            # Stress classification
            if moisture > 0.30:
                stress = "Good"
                flag = "✅"
            elif moisture > 0.22:
                stress = "Adequate"
                flag = "✅"
            elif moisture > 0.15:
                stress = "Moderate Stress"
                flag = "⚠️"
            else:
                stress = "Severe Stress"
                flag = "🔴"

            forecast.append({
                "date": date,
                "day": i + 1,
                "rain_mm": round(rain_d, 1),
                "et0_mm": round(et0_d, 1),
                "moisture_estimate": round(moisture, 3),
                "stress_level": stress,
                "flag": flag
            })

        # Find first stress day
        first_stress = next(
            (f for f in forecast
             if "Stress" in f["stress_level"]), None)

        # Irrigation recommendation
        irrigation_required = any(
            f["stress_level"] == "Severe Stress" for f in forecast[:7])
        irrigation_day = next(
            (f["day"] for f in forecast
             if f["stress_level"] == "Severe Stress"), None)

        return {
            "current_moisture": smap_current,
            "current_status": (
                "Good" if smap_current and smap_current > 0.25 else
                "Adequate" if smap_current and smap_current > 0.18 else
                "Stress" if smap_current else "Unknown"
            ),
            "10_day_forecast": forecast,
            "first_stress_day": first_stress,
            "irrigation_required": irrigation_required,
            "irrigation_recommended_day": irrigation_day,
            "summary": (
                f"Irrigation likely required by Day {irrigation_day}"
                if irrigation_required else
                "Adequate moisture forecast for next 10 days"
            )
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# COMBINED FIELD INTELLIGENCE
# ─────────────────────────────────────────────

def get_field_intelligence(cgm_result, lat, lng):
    """
    Adds all five intelligence features to CGM output
    """
    ci = cgm_result.get("crop_intelligence", {})
    opt = cgm_result.get("optical_data", {})
    phy = cgm_result.get("physiological", {})
    wx = cgm_result.get("weather", {})

    crop_type = ci.get("crop_type")
    current_stage = ci.get("current_stage")
    sowing_date = ci.get("planting_date")
    base_yield = ci.get("yield_estimate_tha")
    ndvi = opt.get("ndvi")
    ndre = opt.get("ndre")
    lai = phy.get("lai_refined") or phy.get("lai_sar")
    canopy = phy.get("canopy_cover_pct")
    avg_temp = wx.get("avg_temp_c")

    # Get recent rainfall and SMAP
    smap = None
    rain_7d = None
    try:
        from datetime import datetime, timedelta
        today = datetime.now()
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lng,
                "start_date": (today - timedelta(days=7)).strftime(
                    "%Y-%m-%d"),
                "end_date": today.strftime("%Y-%m-%d"),
                "daily": "precipitation_sum",
                "timezone": "Europe/Dublin"
            }, timeout=15)
        if r.status_code == 200:
            rain_data = r.json().get("daily", {}).get(
                "precipitation_sum", [])
            rain_7d = round(sum([x for x in rain_data if x]), 1)
    except:
        pass

    # Feature 1 — Risk Score
    risk = calculate_field_risk_score(
        ndvi, ndre, smap, rain_7d, avg_temp,
        crop_type, current_stage, lai, canopy)

    # Feature 2 — Yield Limiting Factors
    yield_factors = calculate_yield_limiting_factors(
        crop_type, ndvi, ndre, smap, canopy, rain_7d, base_yield)

    # Feature 3 — Stage Timeline
    timeline = build_stage_timeline(
        crop_type, sowing_date, current_stage)

    # Feature 4 — Disease Risk
    disease = calculate_disease_risk(
        crop_type, current_stage, avg_temp, rain_7d,
        smap_moisture=smap)

    # Feature 5 — Soil Moisture Profile + Forecast
    from models.soil_moisture import get_soil_moisture_profile
    import json
    try:
        sar_file = "/workspaces/crop-trajectory/sar_ireland_2026.json"
        with open(sar_file) as f:
            sar_obs = json.load(f)
    except:
        sar_obs = None
    
    moisture_forecast = get_soil_moisture_profile(
        lat, lng, crop_type, sar_obs, smap)

    return {
        "field_risk_score": risk,
        "yield_analysis": yield_factors,
        "stage_timeline": timeline,
        "disease_risk": disease,
        "moisture_forecast": moisture_forecast
    }


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    from models.cgm import run_cgm

    lat, lng = 53.6529, -6.6789

    print("Running full CGM + Field Intelligence...")
    cgm = run_cgm(
        lat, lng,
        os.environ.get("CDSE_CLIENT_ID"),
        os.environ.get("CDSE_CLIENT_SECRET")
    )

    print("\nAdding field intelligence features...")
    intel = get_field_intelligence(cgm, lat, lng)

    print("\n" + "="*55)
    print("FIELD INTELLIGENCE REPORT")
    print("="*55)

    # Risk Score
    rs = intel["field_risk_score"]
    print(f"\n📊 Field Health Score: {rs['field_health_score']}/10")
    print(f"   Risk Level: {rs['risk_level']}")
    print(f"   Disease Risk: {rs['sub_scores']['disease_risk']}")
    print(f"   Moisture Stress: {rs['sub_scores']['moisture_stress']}")
    print(f"   Nitrogen: {rs['sub_scores']['nitrogen_deficiency']}")
    print(f"   Yield Risk: {rs['yield_risk']}")

    # Yield Factors
    yf = intel["yield_analysis"]
    print(f"\n🌾 Yield Analysis:")
    print(f"   Base estimate: {yf['base_yield_tha']} t/ha")
    print(f"   Adjusted: {yf['adjusted_yield_tha']} t/ha")
    print(f"   Total impact: {yf['total_yield_impact_pct']}%")
    if yf["limiting_factors"]:
        print("   Limiting factors:")
        for f in yf["limiting_factors"]:
            print(f"     {f['factor']}: {f['yield_impact_pct']}%")

    # Timeline
    tl = intel["stage_timeline"]
    if tl:
        print(f"\n📅 Crop Stage Timeline:")
        for s in tl["stages"]:
            icon = "✅" if s["status"] == "complete" else \
                   "▶️ " if s["status"] == "current" else "⏳"
            days = f"({abs(s['days_from_today'])}d ago)" \
                   if s['days_from_today'] < 0 else \
                   f"(in {s['days_from_today']}d)"
            print(f"   {icon} {s['stage']}: {s['date']} {days}")
        if tl["harvest_window"]:
            print(f"   🚜 Harvest window: {tl['harvest_window']}")

    # Disease Risk
    dr = intel["disease_risk"]
    print(f"\n🦠 Disease Risk:")
    for d in dr["disease_risks"]:
        icon = "🔴" if d["risk_level"] == "High" else \
               "🟡" if d["risk_level"] == "Moderate" else "🟢"
        print(f"   {icon} {d['disease']}: {d['risk_level']}")
        if d["risk_level"] in ["High", "Moderate"]:
            print(f"      → {d['recommended_action']}")

    # Moisture Forecast
    mf = intel["moisture_forecast"]
    if mf and not mf.get("error"):
        print(f"\n💧 Moisture Forecast:")
        print(f"   Current: {mf['current_status']}")
        print(f"   Summary: {mf['summary']}")
        print("   10-day outlook:")
        for f in mf["10_day_forecast"][:5]:
            print(f"   Day {f['day']} ({f['date']}): "
                  f"{f['flag']} {f['stress_level']} "
                  f"(rain:{f['rain_mm']}mm)")
