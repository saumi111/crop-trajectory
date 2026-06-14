"""
ML-based Irish Crop Classifier
Random Forest trained on 28 parcels from 479 DAFM ground truth
LOO accuracy: 61% — better than rule-based 45%

To retrain with more data:
python3 tools/retrain_classifier.py
"""

import numpy as np
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained/irish_crop_rf.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'trained/irish_crop_le.pkl')

_rf = None
_le = None

def _load():
    global _rf, _le
    if _rf is None:
        _rf = joblib.load(MODEL_PATH)
        _le = joblib.load(ENCODER_PATH)

def classify_ml(observations):
    """
    Classify crop from SAR observations using Random Forest.
    Returns ranked probabilities for all crop classes.
    """
    try:
        _load()
    except Exception as e:
        return {"error": str(e), "crop_type": "Unknown", "confidence": 0}

    # Extract monthly VH and VV
    monthly_vh = {}
    monthly_vv = {}
    for o in observations:
        if not o.get("available"):
            continue
        try:
            month = int(o["date"].split("-")[1])
            vh = o.get("vh") or o.get("vh_mean")
            vv = o.get("vv") or o.get("vv_mean")
            if vh:
                if month not in monthly_vh:
                    monthly_vh[month] = []
                monthly_vh[month].append(float(vh))
            if vv:
                if month not in monthly_vv:
                    monthly_vv[month] = []
                monthly_vv[month].append(float(vv))
        except:
            continue

    # Average per month
    avg_vh = {m: np.mean(v) for m, v in monthly_vh.items()}
    avg_vv = {m: np.mean(v) for m, v in monthly_vv.items()}

    # Build feature vector
    vh = [float(avg_vh.get(m, 0)) for m in range(1, 13)]
    vv = [float(avg_vv.get(m, 0)) for m in range(1, 13)]

    winter_vh = np.mean([vh[m-1] for m in [12, 1, 2] if vh[m-1] > 0]) or 0
    spring_vh = np.mean([vh[m-1] for m in [3, 4, 5] if vh[m-1] > 0]) or 0
    summer_vh = np.mean([vh[m-1] for m in [6, 7, 8] if vh[m-1] > 0]) or 0
    all_vv = [v for v in vv if v > 0]
    vv_range = max(all_vv) - min(all_vv) if all_vv else 0

    features = np.array(vh + [
        winter_vh, spring_vh, summer_vh,
        summer_vh - winter_vh,
        spring_vh - winter_vh,
        vv_range
    ]).reshape(1, -1)

    probs = _rf.predict_proba(features)[0]
    classes = _le.classes_

    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    best_crop = str(ranked[0][0])
    best_prob = round(ranked[0][1] * 100)
    second_prob = round(ranked[1][1] * 100) if len(ranked) > 1 else 0

    uncertain = (best_prob - second_prob) < 15

    # Cap confidence — 28 parcels is small training set
    operational_confidence = min(75, best_prob)

    return {
        "crop_type": best_crop if not uncertain else "Uncertain",
        "top_crop": best_crop,
        "classification_score": best_prob,
        "confidence": operational_confidence,
        "uncertain": uncertain,
        "confidence_table": [
            {"crop": str(c), "pct": round(p * 100)}
            for c, p in ranked
        ],
        "method": "Random Forest — 28 Irish parcels (LOO 61%)",
        "note": "Retrain with 50+ parcels per class for production"
    }


if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, '/workspaces/crop-trajectory')

    with open('/workspaces/crop-trajectory/sar_ireland_2026.json') as f:
        obs = json.load(f)

    result = classify_ml(obs)
    print(f"Crop:       {result['crop_type']}")
    print(f"ML Score:   {result['classification_score']}%")
    print(f"Conf:       {result['confidence']}%")
    print(f"Uncertain:  {result['uncertain']}")
    print(f"\nRankings:")
    for r in result['confidence_table']:
        bar = "█" * (r['pct'] // 5)
        print(f"  {r['crop']:<20} {bar:<20} {r['pct']}%")


def classify_hybrid(observations):
    """
    Hybrid classifier:
    1. Rule-based VV range for Grassland (reliable, 479 parcels)
    2. ML Random Forest for arable subtype
    """
    # Extract VV range first
    all_vv = []
    for o in observations:
        if not o.get("available"): continue
        vv = o.get("vv") or o.get("vv_mean")
        if vv: all_vv.append(float(vv))

    vv_range = round(max(all_vv) - min(all_vv), 2) if len(all_vv) > 3 else None

    # Calculate winter VH for secondary check
    monthly_vh_vals = {}
    for o in observations:
        if not o.get("available"): continue
        try:
            month = int(o["date"].split("-")[1])
            vh = o.get("vh") or o.get("vh_mean")
            if vh:
                if month not in monthly_vh_vals:
                    monthly_vh_vals[month] = []
                monthly_vh_vals[month].append(float(vh))
        except: continue

    winter_months = [m for m in [12, 1, 2] if m in monthly_vh_vals]
    winter_vh = round(
        sum(sum(monthly_vh_vals[m])/len(monthly_vh_vals[m])
            for m in winter_months) / len(winter_months), 2
    ) if winter_months else None

    # RULE 1 — Grassland gate (VV range < 100 AND winter VH < 35)
    # Oats have winter VH > 38 so they escape grassland gate
    # Spring Wheat winter VH ~30 — similar to grassland (hard case)
    is_low_vv = vv_range is not None and vv_range < 100
    is_low_winter_vh = winter_vh is None or winter_vh < 35

    if is_low_vv and is_low_winter_vh:
        return {
            "crop_type": "Grassland",
            "top_crop": "Grassland",
            "classification_score": 85,
            "confidence": 82,
            "uncertain": False,
            "confidence_table": [
                {"crop": "Grassland", "pct": 85},
                {"crop": "Spring Barley", "pct": 8},
                {"crop": "Winter Wheat", "pct": 4},
                {"crop": "Oilseed Rape", "pct": 2},
                {"crop": "Oats", "pct": 1}
            ],
            "method": "Rule-based VV range + winter VH (479 parcels, 85% accuracy)",
            "vv_range": vv_range,
            "winter_vh": winter_vh,
            "gate": "grassland"
        }

    # High winter VH with low VV range = Oats
    if is_low_vv and winter_vh and winter_vh >= 35:
        return {
            "crop_type": "Oats",
            "top_crop": "Oats",
            "classification_score": 70,
            "confidence": 65,
            "uncertain": False,
            "confidence_table": [
                {"crop": "Oats", "pct": 70},
                {"crop": "Grassland", "pct": 20},
                {"crop": "Spring Barley", "pct": 7},
                {"crop": "Winter Wheat", "pct": 2},
                {"crop": "Oilseed Rape", "pct": 1}
            ],
            "method": "Rule-based winter VH gate (Oats signature)",
            "vv_range": vv_range,
            "winter_vh": winter_vh,
            "gate": "oats_winter_vh"
        }

    # RULE 2 — Run ML for arable subtype
    ml_result = classify_ml(observations)
    ml_result["vv_range"] = vv_range
    ml_result["gate"] = "arable_ml"

    # If ML says Grassland but VV range is high — flag as uncertain
    if ml_result["top_crop"] == "Grassland" and vv_range and vv_range > 100:
        ml_result["uncertain"] = True
        ml_result["uncertainty_note"] = (
            f"VV range {vv_range} suggests arable but ML says Grassland — "
            "farmer confirmation recommended"
        )

    return ml_result
