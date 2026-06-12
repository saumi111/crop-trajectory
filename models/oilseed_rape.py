"""
Oilseed Rape (OSR) Growth Stage Model
Detects phenological stages from Sentinel-1 SAR time series

Key signatures:
- Sowing (Aug-Sep): soil disturbance signal
- Establishment (Sep-Oct): VH rising slowly
- Rosette (Nov-Feb): low stable canopy
- Stem Extension (Mar): VH rising
- Flowering (Apr-May): VH peak — bright yellow canopy
  SAR: VH maximum, VH/VV ratio peak
- Pod Fill (May-Jun): signal declining
- Ripening (Jun-Jul): sharp VH decline
- Harvest (Jul-Aug): sharp drop both channels

Irish OSR calendar
"""

import numpy as np
from datetime import datetime, timedelta


OSR_STAGES = {
    0:  "Sowing / Bare Soil",
    10: "Emergence / Establishment",
    20: "Rosette",
    30: "Stem Extension",
    60: "Flowering",
    70: "Pod Development",
    80: "Pod Fill / Ripening",
    99: "Harvest"
}

# Typical Irish OSR calendar
# Days from sowing (August sowing assumed)
IRELAND_OSR_TIMELINE = {
    0:  0,   # Sowing — Aug
    10: 21,  # Emergence — Sep
    20: 60,  # Rosette — Oct/Nov
    30: 180, # Stem Extension — Feb/Mar
    60: 240, # Flowering — Apr/May
    70: 270, # Pod Development — May/Jun
    80: 300, # Pod Fill — Jun/Jul
    99: 330  # Harvest — Jul/Aug
}


def calculate_vh_vv_ratio(vv_series, vh_series):
    ratios = []
    for vv, vh in zip(vv_series, vh_series):
        if vv and vh and vv > 0:
            ratios.append(round(vh / vv, 4))
        else:
            ratios.append(None)
    return ratios


def smooth_series(values, window=2):
    smoothed = []
    for i in range(len(values)):
        valid = [v for v in values[max(0,i-window):i+window+1] if v]
        smoothed.append(round(sum(valid)/len(valid), 4) if valid else None)
    return smoothed


def detect_sowing_osr(vv_series, vh_series, dates):
    """
    Detect OSR sowing date
    Sowing August-September in Ireland
    Signature: VV disturbance then rising VH
    """
    for i in range(1, len(vv_series)):
        if dates[i].month not in [8, 9]:
            continue
        if vv_series[i] and vv_series[i-1]:
            drop = vv_series[i] / vv_series[i-1]
            if drop < 0.85:
                return dates[i], dates[i] + timedelta(days=21)

    # Default to first August observation
    for i, date in enumerate(dates):
        if date.month == 8:
            return date, date + timedelta(days=21)

    return None, None


def detect_flowering_osr(vv_series, vh_series, dates, sowing_date=None):
    """
    Detect flowering date
    OSR flowering produces distinctive SAR signature
    VH reaches maximum — dense canopy with flowers
    Occurs April-May in Ireland
    Minimum 220 days after August sowing
    """
    ratios = calculate_vh_vv_ratio(vv_series, vh_series)
    smoothed = smooth_series(ratios)

    peak_val = None
    peak_date = None

    for i, (val, date) in enumerate(zip(smoothed, dates)):
        if val is None:
            continue
        if date.month not in [4, 5]:
            continue
        if sowing_date:
            if (date - sowing_date).days < 200:
                continue
        if peak_val is None or val > peak_val:
            peak_val = val
            peak_date = date

    return peak_date, peak_val


def detect_ripening_osr(vh_series, dates, flowering_date=None):
    """
    Detect ripening — VH declining after flowering peak
    Pod fill and desiccation reduce canopy signal
    """
    if not flowering_date:
        return None

    flower_idx = None
    for i, date in enumerate(dates):
        if date == flowering_date:
            flower_idx = i
            break

    if flower_idx is None:
        return None

    # Look for consistent VH decline after flowering
    for i in range(flower_idx + 1, len(vh_series)):
        if vh_series[i] and vh_series[flower_idx]:
            decline = (vh_series[flower_idx] - vh_series[i]) / \
                      vh_series[flower_idx]
            if decline > 0.20:  # 20% decline from peak
                return dates[i]

    return None


def detect_harvest_osr(vv_series, dates, sowing_date=None):
    """
    Detect OSR harvest — sharp drop both channels
    July-August Ireland
    Minimum 300 days after sowing
    """
    for i in range(1, len(vv_series)):
        if dates[i].month not in [7, 8]:
            continue
        if sowing_date:
            if (dates[i] - sowing_date).days < 290:
                continue
        if vv_series[i] and vv_series[i-1]:
            drop = vv_series[i] / vv_series[i-1]
            if drop < 0.75:
                return dates[i]
    return None


def estimate_osr_stage(observations):
    """
    Main function — estimates current OSR growth stage
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
    sowing_date, emergence_date = detect_sowing_osr(
        vv_series, vh_series, dates)
    flowering_date, flowering_val = detect_flowering_osr(
        vv_series, vh_series, dates, sowing_date)
    ripening_date = detect_ripening_osr(
        vh_series, dates, flowering_date)
    harvest_date = detect_harvest_osr(
        vv_series, dates, sowing_date)

    # Current context
    latest_date = dates[-1]
    days_since_sowing = (latest_date - sowing_date).days \
        if sowing_date else None

    # Determine current stage
    current_stage_id = 0
    current_stage_name = OSR_STAGES[0]

    if sowing_date and days_since_sowing:
        for stage_id, days in sorted(
                IRELAND_OSR_TIMELINE.items(), reverse=True):
            if days_since_sowing >= days:
                current_stage_id = stage_id
                current_stage_name = OSR_STAGES[stage_id]
                break

    # Override with detected events
    if harvest_date and latest_date >= harvest_date:
        current_stage_id = 99
        current_stage_name = OSR_STAGES[99]
    elif ripening_date and latest_date >= ripening_date:
        if 80 > current_stage_id:
            current_stage_id = 80
            current_stage_name = OSR_STAGES[80]
    elif flowering_date and latest_date >= flowering_date:
        if 60 > current_stage_id:
            current_stage_id = 60
            current_stage_name = OSR_STAGES[60]

    # Next stage
    stage_list = sorted(IRELAND_OSR_TIMELINE.keys())
    current_idx = stage_list.index(current_stage_id) \
        if current_stage_id in stage_list else 0
    next_stage_id = None
    next_stage_name = None
    days_to_next = None

    if current_idx < len(stage_list) - 1:
        next_stage_id = stage_list[current_idx + 1]
        next_stage_name = OSR_STAGES[next_stage_id]
        if sowing_date and days_since_sowing is not None:
            days_to_next = max(
                0, IRELAND_OSR_TIMELINE[next_stage_id] - days_since_sowing)

    # Management alerts
    alerts = []
    if current_stage_id == 30:
        alerts.append("Stem extension — assess canopy for sclerotinia risk")
        alerts.append("Light leaf spot monitoring required")
    if current_stage_id == 60:
        alerts.append("Flowering — sclerotinia fungicide timing critical")
        alerts.append("Assess pod set and stem strength")
    if current_stage_id == 80:
        alerts.append("Pod fill — monitor for pod shatter risk")
        alerts.append("Desiccation timing assessment required")

    confidence = "HIGH" if flowering_date else \
                 "MEDIUM" if sowing_date else "LOW"

    return {
        "crop": "Oilseed Rape",
        "location": "Ireland",
        "latest_observation": latest_date.strftime("%Y-%m-%d"),
        "current_stage_id": current_stage_id,
        "current_stage": current_stage_name,
        "next_stage_id": next_stage_id,
        "next_stage": next_stage_name,
        "days_to_next_stage": days_to_next,
        "sowing_date_detected": sowing_date.strftime("%Y-%m-%d") \
            if sowing_date else None,
        "flowering_date_detected": flowering_date.strftime("%Y-%m-%d") \
            if flowering_date else None,
        "ripening_date_detected": ripening_date.strftime("%Y-%m-%d") \
            if ripening_date else None,
        "harvest_date_detected": harvest_date.strftime("%Y-%m-%d") \
            if harvest_date else None,
        "days_since_sowing": days_since_sowing,
        "management_alerts": alerts,
        "confidence": confidence
    }


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    print("Testing Oilseed Rape Growth Model — Ireland")
    print("="*50)

    test_file = "/workspaces/crop-trajectory/sar_ireland_2026.json"

    if os.path.exists(test_file):
        with open(test_file) as f:
            observations = json.load(f)

        result = estimate_osr_stage(observations)

        print(f"Crop:              {result['crop']}")
        print(f"Latest obs:        {result['latest_observation']}")
        print(f"Current stage:     {result['current_stage_id']} "
              f"— {result['current_stage']}")
        print(f"Next stage:        {result['next_stage_id']} "
              f"— {result['next_stage']}")
        print(f"Days to next:      {result['days_to_next_stage']}")
        print(f"Sowing detected:   {result['sowing_date_detected']}")
        print(f"Flowering detected:{result['flowering_date_detected']}")
        print(f"Ripening detected: {result['ripening_date_detected']}")
        print(f"Days since sowing: {result['days_since_sowing']}")
        print(f"Confidence:        {result['confidence']}")
        if result.get('management_alerts'):
            print("Management alerts:")
            for alert in result['management_alerts']:
                print(f"  → {alert}")
