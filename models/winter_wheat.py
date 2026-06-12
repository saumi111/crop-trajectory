"""
Winter Wheat Growth Stage Model
Detects BBCH growth stages from Sentinel-1 SAR time series

Key signatures:
- Sowing (Oct-Nov): VV drop after soil disturbance
- Vernalisation (Dec-Feb): low stable signal
- Tillering (Mar): VH starts rising
- Stem Extension (Apr-May): VH peak building
- Heading (Jun): sharp VV rise 8dB spike
- Grain Fill (Jul): signal declining
- Harvest (Aug): sharp drop to baseline

Irish Winter Wheat calendar based on typical season
"""

import numpy as np
from datetime import datetime, timedelta


WINTER_WHEAT_STAGES = {
    0:  "Sowing / Bare Soil",
    10: "Emergence",
    20: "Tillering",
    30: "Stem Extension",
    49: "Booting",
    55: "Heading / Ear Emergence",
    73: "Grain Fill",
    87: "Ripening",
    99: "Harvest / Post-Harvest"
}

# Typical Irish Winter Wheat calendar
# Days from sowing (October sowing assumed)
IRELAND_WHEAT_TIMELINE = {
    0:  0,   # Sowing — Oct
    10: 21,  # Emergence — Oct/Nov
    20: 60,  # Tillering — Dec
    30: 180, # Stem Extension — Mar/Apr
    49: 210, # Booting — Apr/May
    55: 240, # Heading — Jun
    73: 270, # Grain Fill — Jul
    87: 290, # Ripening — Jul/Aug
    99: 310  # Harvest — Aug
}


def calculate_vh_vv_ratio(vv_series, vh_series):
    ratios = []
    for vv, vh in zip(vv_series, vh_series):
        if vv and vh and vv > 0:
            ratios.append(round(vh / vv, 4))
        else:
            ratios.append(None)
    return ratios


def calculate_velocity(values, dates):
    velocities = [None]
    for i in range(1, len(values)):
        if values[i] and values[i-1]:
            days = (dates[i] - dates[i-1]).days
            if days > 0:
                vel = (values[i] - values[i-1]) / days
                velocities.append(round(vel, 6))
            else:
                velocities.append(None)
        else:
            velocities.append(None)
    return velocities


def detect_sowing_winter(vv_series, vh_series, dates):
    """
    Detect winter wheat sowing date
    Signature: VV drop after soil disturbance in Oct-Nov
    followed by low stable period
    """
    for i in range(1, len(vv_series)):
        if vv_series[i] and vv_series[i-1]:
            # Only look October-November
            if dates[i].month not in [10, 11]:
                continue
            # VV drop after tillage
            drop_ratio = vv_series[i] / vv_series[i-1]
            if drop_ratio < 0.85:
                sowing_date = dates[i]
                emergence_date = dates[i] + timedelta(days=21)
                return sowing_date, emergence_date

    # If no drop detected estimate from first Oct observation
    for i, date in enumerate(dates):
        if date.month == 10:
            return date, date + timedelta(days=21)

    return None, None


def detect_heading_wheat(vv_series, vh_series, dates, sowing_date=None):
    """
    Detect heading date
    Signature: Sharp VV rise in June
    Minimum 220 days after October sowing
    """
    for i in range(1, len(vv_series)):
        if vv_series[i] and vv_series[i-1]:

            # Only look May-July for Ireland
            if dates[i].month not in [5, 6, 7]:
                continue

            # Minimum days from sowing
            if sowing_date:
                days_from_sowing = (dates[i] - sowing_date).days
                if days_from_sowing < 200:
                    continue

            rise_ratio = vv_series[i] / vv_series[i-1] \
                if vv_series[i-1] > 0 else 0
            if rise_ratio > 1.3:
                return dates[i], rise_ratio

    return None, None


def detect_harvest_wheat(vv_series, dates, sowing_date=None):
    """
    Detect harvest date
    Signature: Sharp VV drop in July-August
    Minimum 280 days after October sowing
    """
    for i in range(1, len(vv_series)):
        if vv_series[i] and vv_series[i-1]:

            # Only July-September
            if dates[i].month not in [7, 8, 9]:
                continue

            if sowing_date:
                days_from_sowing = (dates[i] - sowing_date).days
                if days_from_sowing < 270:
                    continue

            drop_ratio = vv_series[i] / vv_series[i-1] \
                if vv_series[i-1] > 0 else 1
            if drop_ratio < 0.75:
                return dates[i]

    return None


def estimate_wheat_stage(observations):
    """
    Main function — estimates current Winter Wheat BBCH stage
    from SAR time series

    Args:
        observations: list of dicts with date, vv, vh, rvi

    Returns:
        dict with current stage, next stage, management alerts
    """
    if not observations:
        return {"error": "No observations provided"}

    dates = []
    vv_series = []
    vh_series = []

    for obs in observations:
        if obs.get("available") and obs.get("vv") and obs.get("vh"):
            dates.append(datetime.strptime(obs["date"], "%Y-%m-%d"))
            vv_series.append(obs["vv"])
            vh_series.append(obs["vh"])

    if len(dates) < 3:
        return {"error": "Insufficient observations"}

    # Detect key events
    sowing_date, emergence_date = detect_sowing_winter(
        vv_series, vh_series, dates)
    heading_date, heading_conf = detect_heading_wheat(
        vv_series, vh_series, dates, sowing_date)
    harvest_date = detect_harvest_wheat(
        vv_series, dates, sowing_date)

    # Current context
    latest_date = dates[-1]
    days_since_sowing = (latest_date - sowing_date).days \
        if sowing_date else None

    # Determine current stage from days
    current_bbch = 0
    current_stage_name = WINTER_WHEAT_STAGES[0]

    if sowing_date and days_since_sowing:
        for bbch, days in sorted(
                IRELAND_WHEAT_TIMELINE.items(), reverse=True):
            if days_since_sowing >= days:
                current_bbch = bbch
                current_stage_name = WINTER_WHEAT_STAGES[bbch]
                break

    # Override with detected events
    if harvest_date and latest_date >= harvest_date:
        if 99 > current_bbch:
            current_bbch = 99
            current_stage_name = WINTER_WHEAT_STAGES[99]
    elif heading_date and latest_date >= heading_date:
        if 55 > current_bbch:
            current_bbch = 55
            current_stage_name = WINTER_WHEAT_STAGES[55]

    # Next stage
    stage_list = sorted(IRELAND_WHEAT_TIMELINE.keys())
    current_idx = stage_list.index(current_bbch) \
        if current_bbch in stage_list else 0
    next_bbch = None
    next_stage_name = None
    days_to_next = None

    if current_idx < len(stage_list) - 1:
        next_bbch = stage_list[current_idx + 1]
        next_stage_name = WINTER_WHEAT_STAGES[next_bbch]
        if sowing_date and days_since_sowing is not None:
            days_to_next_from_sowing = IRELAND_WHEAT_TIMELINE[next_bbch]
            days_to_next = max(0, days_to_next_from_sowing - days_since_sowing)

    # Management alerts
    alerts = []
    if current_bbch == 30:
        alerts.append("Stem extension — T1 fungicide timing window")
        alerts.append("First nitrogen split application due")
    if current_bbch == 49:
        alerts.append("Booting — T2 fungicide timing window opening")
        alerts.append("Flag leaf protection critical for yield")
    if current_bbch == 55:
        alerts.append("Heading — T3 ear wash fungicide if required")
        alerts.append("Final nitrogen if not yet applied")
    if current_bbch == 73:
        alerts.append("Grain fill — protect canopy")
        alerts.append("Monitor for foliar disease")

    confidence = "HIGH" if heading_date else \
                 "MEDIUM" if sowing_date else "LOW"

    return {
        "crop": "Winter Wheat",
        "location": "Ireland",
        "latest_observation": latest_date.strftime("%Y-%m-%d"),
        "current_bbch": current_bbch,
        "current_stage": current_stage_name,
        "next_bbch": next_bbch,
        "next_stage": next_stage_name,
        "days_to_next_stage": days_to_next,
        "sowing_date_detected": sowing_date.strftime("%Y-%m-%d") \
            if sowing_date else None,
        "heading_date_detected": heading_date.strftime("%Y-%m-%d") \
            if heading_date else None,
        "harvest_date_detected": harvest_date.strftime("%Y-%m-%d") \
            if harvest_date else None,
        "days_since_sowing": days_since_sowing,
        "management_alerts": alerts,
        "confidence": confidence
    }


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    print("Testing Winter Wheat Growth Model — Ireland")
    print("="*50)

    # Use 2026 dataset which covers Oct 2025 to Jun 2026
    test_file = "/workspaces/crop-trajectory/sar_ireland_2026.json"

    if os.path.exists(test_file):
        with open(test_file) as f:
            observations = json.load(f)

        result = estimate_wheat_stage(observations)

        print(f"Crop:              {result['crop']}")
        print(f"Latest obs:        {result['latest_observation']}")
        print(f"Current stage:     BBCH {result['current_bbch']} "
              f"— {result['current_stage']}")
        print(f"Next stage:        BBCH {result['next_bbch']} "
              f"— {result['next_stage']}")
        print(f"Days to next:      {result['days_to_next_stage']}")
        print(f"Sowing detected:   {result['sowing_date_detected']}")
        print(f"Heading detected:  {result['heading_date_detected']}")
        print(f"Harvest detected:  {result['harvest_date_detected']}")
        print(f"Days since sowing: {result['days_since_sowing']}")
        print(f"Confidence:        {result['confidence']}")
        if result.get('management_alerts'):
            print("Management alerts:")
            for alert in result['management_alerts']:
                print(f"  → {alert}")
