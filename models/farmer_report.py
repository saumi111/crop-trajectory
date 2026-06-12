"""
Farmer-Friendly Report Generator
Translates technical CGM output into plain English
decisions that farmers understand immediately.
"""

from datetime import datetime


def translate_weather(gdd, rainfall, avg_temp, water_balance):
    """Translate weather numbers into plain English"""

    temp_status = (
        "Warm — suitable for crop development"
        if avg_temp and avg_temp > 14 else
        "Cool — slower development expected"
        if avg_temp and avg_temp < 10 else
        "Suitable for current growth stage"
        if avg_temp else "Data unavailable"
    )

    rain_status = (
        "Adequate" if rainfall and 400 <= rainfall <= 900 else
        "Dry — monitor soil moisture" if rainfall and rainfall < 400 else
        "High — waterlogging risk" if rainfall and rainfall > 900 else
        "Unknown"
    )

    # Growth status with consistent explanation
    if gdd and gdd > 2000:
        growth_status = "Ahead of normal season"
        growth_reason = (
            "High heat accumulation earlier in season "
            "accelerated development despite current cool temperatures"
            if avg_temp and avg_temp < 12 else
            "Warm season has accelerated crop development"
        )
    elif gdd and gdd > 1500:
        growth_status = "Normal"
        growth_reason = "Crop developing at expected seasonal rate"
    elif gdd and gdd > 1000:
        growth_status = "Slightly behind normal"
        growth_reason = "Cool temperatures have slowed development"
    else:
        growth_status = "Early season"
        growth_reason = "Insufficient heat units accumulated"

    water_comment = (
        "Good water surplus" if water_balance and water_balance > 300 else
        "Adequate" if water_balance and water_balance > 0 else
        "Deficit — irrigation may be needed" if water_balance else "Unknown"
    )

    return {
        "temperature": temp_status,
        "rainfall": rain_status,
        "growth_status": growth_status,
        "growth_reason": growth_reason,
        "water_balance": water_comment
    }


def calculate_yield_confidence(cgm_result, field_intel):
    """
    Calculate yield forecast confidence
    Based on multiple factors
    """
    score = 100
    reasons_good = []
    reasons_concern = []

    ci = cgm_result.get("crop_intelligence", {})
    opt = cgm_result.get("optical_data", {})
    phy = cgm_result.get("physiological", {})

    # Classification confidence
    conf = ci.get("classification_confidence", 0)
    if conf < 50:
        score -= 20
        reasons_concern.append("Low crop classification confidence")
    else:
        reasons_good.append("Good crop identification")

    # NDVI
    ndvi = opt.get("ndvi")
    if ndvi and ndvi > 0.6:
        reasons_good.append("Good vegetation status")
    elif ndvi and ndvi < 0.4:
        score -= 15
        reasons_concern.append("Weak canopy signal")

    # Nitrogen
    nitrogen = opt.get("nitrogen_status")
    if nitrogen == "adequate":
        reasons_good.append("Adequate nitrogen")
    elif nitrogen == "low":
        score -= 10
        reasons_concern.append("Possible nitrogen limitation")

    # Biomass
    biomass = phy.get("biomass_t_ha")
    if biomass and biomass > 5:
        reasons_good.append("Adequate biomass")
    elif biomass and biomass < 3:
        score -= 10
        reasons_concern.append("Low biomass")

    # Canopy cover
    canopy = phy.get("canopy_cover_pct")
    if canopy and canopy > 70:
        reasons_good.append("Good canopy cover")
    elif canopy and canopy < 50:
        score -= 10
        reasons_concern.append("Low canopy cover")

    score = max(0, min(100, score))
    label = "High" if score >= 75 else "Medium" if score >= 55 else "Low"

    return {
        "confidence_pct": score,
        "confidence_label": label,
        "supporting_factors": reasons_good,
        "risk_factors": reasons_concern
    }


def generate_weekly_actions(crop_type, current_stage,
                             disease_risks, moisture_forecast,
                             nitrogen_status, yield_risk):
    """
    Generate THIS WEEK'S ACTIONS
    Prioritised, plain English recommendations
    """
    actions = []
    priority = "Low"

    # Disease actions
    if disease_risks:
        high_risks = [d for d in disease_risks
                      if d["risk_level"] == "High"]
        mod_risks = [d for d in disease_risks
                     if d["risk_level"] == "Moderate"]

        if high_risks:
            priority = "High"
            for d in high_risks:
                actions.append({
                    "priority": "Urgent",
                    "action": d["recommended_action"],
                    "reason": f"{d['disease']} risk is HIGH"
                })
        elif mod_risks:
            if priority != "High":
                priority = "Medium"
            for d in mod_risks[:2]:
                actions.append({
                    "priority": "Monitor",
                    "action": d["recommended_action"],
                    "reason": f"{d['disease']} risk is Moderate"
                })

    # Nitrogen actions
    if nitrogen_status == "low":
        priority = "High" if priority != "High" else priority
        actions.append({
            "priority": "Action",
            "action": "Consider foliar nitrogen application",
            "reason": "Nitrogen levels below optimum"
        })
    elif nitrogen_status == "adequate":
        actions.append({
            "priority": "Info",
            "action": "No additional nitrogen required",
            "reason": "Nitrogen levels adequate"
        })

    # Moisture actions
    if moisture_forecast:
        if moisture_forecast.get("irrigation_required"):
            day = moisture_forecast.get("irrigation_recommended_day")
            priority = "High" if priority != "High" else priority
            actions.append({
                "priority": "Urgent",
                "action": f"Plan irrigation — moisture stress expected Day {day}",
                "reason": "Soil moisture forecast below threshold"
            })
        else:
            actions.append({
                "priority": "Info",
                "action": "Continue monitoring soil moisture",
                "reason": "Adequate moisture forecast next 10 days"
            })

    # Yield monitoring
    if yield_risk == "High":
        actions.append({
            "priority": "Monitor",
            "action": "Yield forecast under pressure — review field conditions",
            "reason": "Multiple yield limiting factors detected"
        })
    else:
        actions.append({
            "priority": "Info",
            "action": "Yield forecast remains stable",
            "reason": "No major limiting factors detected"
        })

    # Canopy monitoring
    actions.append({
        "priority": "Info",
        "action": "Continue monitoring canopy development",
        "reason": f"Crop at {current_stage} stage"
    })

    # Overall assessment
    urgent_actions = [a for a in actions if a["priority"] == "Urgent"]
    monitor_actions = [a for a in actions if a["priority"] == "Monitor"]

    if not urgent_actions and not monitor_actions:
        assessment = "No urgent intervention required"
    elif urgent_actions and len(urgent_actions) >= 2:
        assessment = "Urgent actions required this week"
    elif urgent_actions:
        assessment = "One urgent action required — monitor closely"
    else:
        assessment = "Monitor closely — some actions recommended"

    # Calculate priority from action severity — not just presence
    urgent_count = sum(1 for a in actions if a["priority"] == "Urgent")
    action_count = sum(1 for a in actions if a["priority"] == "Action")
    monitor_count = sum(1 for a in actions if a["priority"] == "Monitor")

    if urgent_count >= 2:
        priority = "High"
    elif urgent_count == 1 and action_count >= 1:
        priority = "High"
    elif urgent_count == 1 or action_count >= 2:
        priority = "Medium"
    elif monitor_count >= 2:
        priority = "Medium"
    else:
        priority = "Low"

    return {
        "priority": priority,
        "actions": actions,
        "overall_assessment": assessment
    }


def generate_farmer_report(cgm_result, field_intel):
    """
    Generate complete farmer-friendly report
    from technical CGM and field intelligence output
    """
    ci = cgm_result.get("crop_intelligence", {})
    opt = cgm_result.get("optical_data", {})
    phy = cgm_result.get("physiological", {})
    wx = cgm_result.get("weather", {})

    crop_type = ci.get("crop_type", "Unknown")
    conf = ci.get("classification_confidence", 0)
    conf_label = ci.get("classification_confidence_label", "Low")
    current_stage = ci.get("current_stage", "Unknown")
    yield_est = ci.get("yield_estimate_tha")
    yield_range = ci.get("yield_range")
    planting = ci.get("planting_date")
    alternatives = ci.get("alternative_matches", [])

    ndre = opt.get("ndvi")
    nitrogen = opt.get("nitrogen_status", "Unknown")
    biomass = phy.get("biomass_t_ha")
    canopy = phy.get("canopy_cover_pct")
    lai = phy.get("lai_refined") or phy.get("lai_sar")

    risk_score = field_intel.get("field_risk_score", {})
    disease = field_intel.get("disease_risk", {})
    moisture = field_intel.get("moisture_forecast", {})
    yield_factors = field_intel.get("yield_analysis", {})
    timeline = field_intel.get("stage_timeline")

    # Translate weather
    weather_plain = translate_weather(
        wx.get("gdd_accumulated_base0"),
        wx.get("total_rainfall_mm"),
        wx.get("avg_temp_c"),
        wx.get("water_balance_mm")
    )

    # Yield confidence
    yield_conf = calculate_yield_confidence(cgm_result, field_intel)

    # Priority action
    disease_risks = disease.get("disease_risks", [])
    priority_disease = next(
        (d for d in disease_risks
         if d["risk_level"] in ["High", "Moderate"]), None)
    priority_action = (
        priority_disease["recommended_action"]
        if priority_disease else
        ci.get("management_alerts", ["Monitor crop development"])[0]
        if ci.get("management_alerts") else
        "Monitor crop development"
    )

    # Next stage
    next_stage_text = None
    if timeline:
        next_s = timeline.get("next_stage")
        if next_s:
            days = next_s.get("days_from_today", 0)
            weeks = days // 7
            next_stage_text = (
                f"{next_s['stage']} "
                f"(approx. {weeks}-{weeks+1} weeks)"
                if weeks > 0 else
                f"{next_s['stage']} (imminent)"
            )

    # Weekly actions
    weekly = generate_weekly_actions(
        crop_type, current_stage,
        disease_risks, moisture,
        nitrogen,
        risk_score.get("yield_risk", "Medium")
    )

    return {
        "farmer_summary": {
            "crop": f"{crop_type} ({conf_label} confidence)",
            "alternative_matches": [
                f"{a['crop']} ({a['pct']}%)"
                for a in alternatives
            ],
            "current_stage": current_stage,
            "planting_date": planting,
            "field_health": f"{risk_score.get('field_health_score', 'N/A')}/10",
            "expected_yield": (
                f"{yield_est} t/ha ({yield_range})"
                if yield_est else "Insufficient data"
            ),
            "yield_confidence": (
                f"{yield_conf['confidence_pct']}% "
                f"({yield_conf['confidence_label']})"
            ),
            "nitrogen_status": nitrogen.capitalize() if nitrogen else "Unknown",
            "biomass": f"{biomass} t/ha DM" if biomass else "Unknown",
            "canopy_cover": f"{canopy}%" if canopy else "Unknown",
            "yield_risk": risk_score.get("yield_risk", "Unknown"),
            "priority_action": priority_action,
            "next_stage": next_stage_text or "Insufficient data"
        },
        "weather_plain": weather_plain,
        "yield_forecast": {
            "estimate_tha": yield_est,
            "range": yield_range,
            "confidence_pct": yield_conf["confidence_pct"],
            "confidence_label": yield_conf["confidence_label"],
            "supporting_factors": yield_conf["supporting_factors"],
            "risk_factors": yield_conf["risk_factors"],
            "limiting_factors": yield_factors.get("limiting_factors", [])
        },
        "this_weeks_actions": weekly,
        "disease_risk": disease,
        "moisture_forecast": moisture
    }


def print_farmer_report(report):
    """Print formatted farmer report"""
    fs = report["farmer_summary"]
    wx = report["weather_plain"]
    yf = report["yield_forecast"]
    wa = report["this_weeks_actions"]

    print("\n" + "="*52)
    print("FIELD SUMMARY")
    print("="*52)
    print(f"\nCrop:             {fs['crop']}")
    if fs['alternative_matches']:
        print(f"Other matches:    {', '.join(fs['alternative_matches'])}")
    print(f"Current Stage:    {fs['current_stage']}")
    if fs['planting_date']:
        print(f"Planting Date:    {fs['planting_date']}")
    print(f"\nField Health:     {fs['field_health']}")
    print(f"Yield Risk:       {fs['yield_risk']}")
    print(f"\nExpected Yield:   {fs['expected_yield']}")
    print(f"Yield Confidence: {fs['yield_confidence']}")
    if yf['supporting_factors']:
        for f in yf['supporting_factors']:
            print(f"  ✓ {f}")
    if yf['risk_factors']:
        for f in yf['risk_factors']:
            print(f"  ⚠ {f}")
    if yf['limiting_factors']:
        print("\nYield Constraints:")
        for f in yf['limiting_factors']:
            print(f"  {f['factor']}: {f['yield_impact_pct']}%")

    print(f"\nNitrogen:         {fs['nitrogen_status']}")
    print(f"Biomass:          {fs['biomass']}")
    print(f"Canopy Cover:     {fs['canopy_cover']}")

    print(f"\nNext Stage:       {fs['next_stage']}")
    print(f"Priority Action:  {fs['priority_action']}")

    print(f"\n--- Weather ---")
    print(f"Temperature:      {wx['temperature']}")
    print(f"Rainfall:         {wx['rainfall']}")
    print(f"Growth Status:    {wx['growth_status']}")
    print(f"Water Balance:    {wx['water_balance']}")

    print(f"\n--- Soil Moisture ---")
    mf = report.get("moisture_forecast", {})
    if mf and not mf.get("error"):
        rz = mf.get("root_zone", {})
        ds = mf.get("data_sources", {})
        if rz:
            print(f"Data source:      {ds.get('root_zone','Water balance model')}")
            print(f"SAR corrected:    {'Yes' if ds.get('sar_corrections_applied') else 'No'}")
            print(f"Root zone:        {rz.get('current_moisture','N/A')} m3/m3")
            print(f"Status:           {rz.get('status_icon','')} {rz.get('status','Unknown')}")
            print(f"Trend:            {rz.get('trend_icon','')} {rz.get('trend','Unknown')}")
            print(f"Recommendation:   {mf.get('irrigation_recommendation','N/A')}")
            history = mf.get("10_day_history", [])[-5:]
            if history:
                print("  Recent history:")
                for h in history:
                    icon = "✅" if h["status"] in ["Optimal","Adequate"] else "⚠️"
                    sar = " SAR✓" if h.get("sar_corrected") else ""
                    print(f"  {h['date']}: {h['root_zone_moisture']} "
                          f"{icon} {h['status']}{sar}")
        else:
            print("  Moisture data unavailable")
    else:
        print("  Moisture data unavailable")

    print(f"THIS WEEK'S ACTIONS  |  Priority: {wa['priority']}")
    print("="*52)
    for i, action in enumerate(wa["actions"], 1):
        icon = "🔴" if action["priority"] == "Urgent" else \
               "🟡" if action["priority"] in ["Action", "Monitor"] else "ℹ️ "
        print(f"\n{i}. {icon} {action['action']}")
        print(f"   Reason: {action['reason']}")
    print(f"\n→ {wa['overall_assessment']}")
    print("="*52)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')
    from models.cgm import run_cgm
    from models.field_intelligence import get_field_intelligence

    lat, lng = 53.6529, -6.6789

    print("Running CGM...")
    cgm = run_cgm(
        lat, lng,
        os.environ.get("CDSE_CLIENT_ID"),
        os.environ.get("CDSE_CLIENT_SECRET")
    )

    print("Adding field intelligence...")
    intel = get_field_intelligence(cgm, lat, lng)

    print("Generating farmer report...")
    report = generate_farmer_report(cgm, intel)
    print_farmer_report(report)
