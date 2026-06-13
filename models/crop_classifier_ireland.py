"""
Irish Crop Classifier — Data-Driven
Built from real DAFM parcel SAR signatures
479 confirmed parcels across Ireland

Replaces the rule-based classifier with
signatures derived from ground truth data

Key discriminating features discovered:
- VV range < 100 → Grassland (low seasonal variation)
- Summer VH < 16 → Spring cereals (harvest drop)
- Winter VH > 38 → Oats (high winter signal)
- Spring VH drop → OSR (flowering stage)
- Summer VH drop from winter → Winter cereals
"""

import numpy as np
from datetime import datetime


# Real Irish SAR signatures from DAFM parcels
# Source: 479 confirmed parcels June 2026
IRISH_SIGNATURES = {
    "Grassland": {
        "winter_vh": 28.60,
        "spring_vh": 24.39,
        "summer_vh": 28.75,
        "vv_range": 71.54,
        "mean_rvi": 0.58,
        "key_feature": "Low VV range — stable year-round signal"
    },
    "Winter Wheat": {
        "winter_vh": 29.77,
        "spring_vh": 24.37,
        "summer_vh": 20.03,
        "vv_range": 189.84,
        "mean_rvi": 0.51,
        "key_feature": "Summer VH drops at harvest"
    },
    "Spring Barley": {
        "winter_vh": 28.55,
        "spring_vh": 25.42,
        "summer_vh": 34.12,
        "vv_range": 227.11,
        "mean_rvi": 0.50,
        "key_feature": "Highest VV range — strong seasonal swing"
    },
    "Oilseed Rape": {
        "winter_vh": 32.62,
        "spring_vh": 23.14,
        "summer_vh": 29.78,
        "vv_range": 213.84,
        "mean_rvi": 0.54,
        "key_feature": "Spring VH dip at flowering then recovery"
    },
    "Oats": {
        "winter_vh": 41.31,
        "spring_vh": 30.03,
        "summer_vh": 52.86,
        "vv_range": 143.67,
        "mean_rvi": 0.58,
        "key_feature": "High winter VH — highest of all crops"
    },
    "Maize": {
        "winter_vh": 31.81,
        "spring_vh": 25.46,
        "summer_vh": 22.48,
        "vv_range": 181.75,
        "mean_rvi": 0.52,
        "key_feature": "Moderate range, summer VH declines"
    },
    "Spring Wheat": {
        "winter_vh": 29.75,
        "spring_vh": 20.25,
        "summer_vh": 14.24,
        "vv_range": 131.21,
        "mean_rvi": 0.55,
        "key_feature": "Lowest summer VH — strong harvest drop"
    }
}


def get_seasonal_stats(observations):
    """Extract seasonal VH and VV statistics"""
    monthly_vh = {}
    monthly_vv = {}
    all_vv = []
    all_rvi = []

    for obs in observations:
        if not obs.get("available"):
            continue
        month = int(obs["date"].split("-")[1])
        vv = obs.get("vv") or obs.get("vv_mean")
        vh = obs.get("vh") or obs.get("vh_mean")
        rvi = obs.get("rvi") or obs.get("rvi_mean")

        if vv:
            all_vv.append(vv)
            if month not in monthly_vv:
                monthly_vv[month] = []
            monthly_vv[month].append(vv)

        if vh:
            if month not in monthly_vh:
                monthly_vh[month] = []
            monthly_vh[month].append(vh)

        if rvi:
            all_rvi.append(rvi)

    def season_avg(months):
        vals = []
        for m in months:
            if m in monthly_vh:
                vals.extend(monthly_vh[m])
        return round(np.mean(vals), 2) if vals else None

    winter_vh = season_avg([12, 1, 2])
    spring_vh = season_avg([3, 4, 5])
    summer_vh = season_avg([6, 7, 8])
    autumn_vh = season_avg([9, 10, 11])

    vv_range = round(max(all_vv) - min(all_vv), 2) if all_vv else 0
    mean_rvi = round(np.mean(all_rvi), 3) if all_rvi else None

    return {
        "winter_vh": winter_vh,
        "spring_vh": spring_vh,
        "summer_vh": summer_vh,
        "autumn_vh": autumn_vh,
        "vv_range": vv_range,
        "mean_rvi": mean_rvi
    }


def classify_ireland(observations):
    """
    Classify Irish crop type using data-driven SAR signatures
    Uses Euclidean distance to nearest known signature

    Returns crop type with confidence score
    """
    stats = get_seasonal_stats(observations)

    if not stats["winter_vh"] and not stats["summer_vh"]:
        return {
            "crop_type": "Unknown",
            "confidence_pct": 0,
            "reason": "Insufficient seasonal data",
            "signature_source": "DAFM ground truth — 479 Irish parcels"
        }

    # Rule-based pre-filter using key features
    # These are the strongest discriminating signals

    # Grassland — VV range is key
    if stats["vv_range"] and stats["vv_range"] < 100:
        return {
            "crop_type": "Grassland",
            "confidence_pct": 85,
            "classification_reasons": [
                f"Low VV range ({stats['vv_range']}) — stable year-round signal",
                "Consistent with permanent pasture signature"
            ],
            "stats": stats,
            "signature_source": "DAFM ground truth — 479 Irish parcels"
        }

    # Oats — very high winter VH
    if stats["winter_vh"] and stats["winter_vh"] > 38:
        return {
            "crop_type": "Oats",
            "confidence_pct": 75,
            "classification_reasons": [
                f"High winter VH ({stats['winter_vh']}) — oats signature",
                "Winter VH exceeds all other Irish crops"
            ],
            "stats": stats,
            "signature_source": "DAFM ground truth — 479 Irish parcels"
        }

    # Spring Wheat — very low summer VH
    if stats["summer_vh"] and stats["summer_vh"] < 16:
        return {
            "crop_type": "Spring Wheat",
            "confidence_pct": 78,
            "classification_reasons": [
                f"Very low summer VH ({stats['summer_vh']}) — harvest drop",
                "Spring wheat shows strongest summer decline"
            ],
            "stats": stats,
            "signature_source": "DAFM ground truth — 479 Irish parcels"
        }

    # Distance-based classification for remaining crops
    scores = {}
    for crop, sig in IRISH_SIGNATURES.items():
        distance = 0
        weight_total = 0

        if stats["winter_vh"] and sig["winter_vh"]:
            distance += abs(stats["winter_vh"] - sig["winter_vh"]) * 1.0
            weight_total += 1

        if stats["spring_vh"] and sig["spring_vh"]:
            distance += abs(stats["spring_vh"] - sig["spring_vh"]) * 1.5
            weight_total += 1.5

        if stats["summer_vh"] and sig["summer_vh"]:
            distance += abs(stats["summer_vh"] - sig["summer_vh"]) * 2.0
            weight_total += 2.0

        if stats["vv_range"] and sig["vv_range"]:
            distance += abs(stats["vv_range"] - sig["vv_range"]) * 0.1
            weight_total += 0.5

        if weight_total > 0:
            scores[crop] = distance / weight_total

    if not scores:
        return {"crop_type": "Unknown", "confidence_pct": 0}

    # Best match = lowest distance
    best_crop = min(scores, key=scores.get)
    best_score = scores[best_crop]

    # Convert distance to confidence
    # Lower distance = higher confidence
    max_dist = max(scores.values())
    if max_dist > 0:
        confidence = round((1 - best_score / max_dist) * 100)
    else:
        confidence = 50

    confidence = max(40, min(95, confidence))

    # Reasons based on key features
    reasons = [
        f"Closest match to Irish {best_crop} SAR signature",
        f"VV range: {stats['vv_range']} (reference: {IRISH_SIGNATURES[best_crop]['vv_range']})",
        IRISH_SIGNATURES[best_crop]["key_feature"]
    ]

    # Alternatives
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    alternatives = [
        {"crop": c, "distance": round(d, 1)}
        for c, d in sorted_scores[1:4]
    ]

    return {
        "crop_type": best_crop,
        "confidence_pct": confidence,
        "classification_reasons": reasons,
        "alternatives": alternatives,
        "stats": stats,
        "signature_source": "DAFM ground truth — 479 Irish parcels"
    }


if __name__ == "__main__":
    import json
    import os
    import sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    os.environ['CDSE_CLIENT_ID'] = \
        'sh-6e5978f5-f5d6-43d6-874d-720d84121683'
    os.environ['CDSE_CLIENT_SECRET'] = \
        'yrMEXQ5drlF26yrB4sTEXfWOIwKtB1fP'

    print("Testing Irish Crop Classifier")
    print("="*50)

    # Test on the Co. Meath grassland parcel we know
    with open('/workspaces/crop-trajectory/sar_ireland_2026.json') as f:
        obs = json.load(f)
    available = [o for o in obs if o.get('available')]

    result = classify_ireland(available)
    print(f"Known crop: Permanent Pasture (DAFM)")
    print(f"Classified: {result['crop_type']}")
    print(f"Confidence: {result['confidence_pct']}%")
    print(f"Reasons:")
    for r in result.get('classification_reasons', []):
        print(f"  → {r}")
    print(f"Source: {result.get('signature_source')}")
