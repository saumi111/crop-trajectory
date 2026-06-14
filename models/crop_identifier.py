"""
Crop Identification from SAR Time Series
Uses VH backscatter seasonal pattern to identify crop type
No reliance on DAFM declaration date

Key signatures (from 479 Irish parcels):
Grassland:    flat VH year-round, VV range < 100
Spring Barley: low winter VH, rising spring, drops at harvest
Winter Wheat:  moderate winter VH, peaks May-Jun, drops Jul
Oilseed Rape:  high winter VH, dips at flowering Mar-Apr
Oats:         highest winter VH of all crops
"""

import numpy as np
from datetime import datetime


def extract_monthly_vh(observations):
    """Extract monthly average VH from SAR time series"""
    monthly = {}
    for o in observations:
        if not o.get("available"):
            continue
        try:
            month = int(o["date"].split("-")[1])
            vh = o.get("vh") or o.get("vh_mean")
            vv = o.get("vv") or o.get("vv_mean")
            if vh:
                if month not in monthly:
                    monthly[month] = {"vh": [], "vv": []}
                monthly[month]["vh"].append(vh)
                if vv:
                    monthly[month]["vv"].append(vv)
        except:
            continue

    result = {}
    for m, vals in monthly.items():
        result[m] = {
            "vh": round(np.mean(vals["vh"]), 2),
            "vv": round(np.mean(vals["vv"]), 2) if vals["vv"] else None
        }
    return result


def analyse_time_series(observations):
    """
    Analyse SAR time series to identify crop type and growth stage.

    Returns:
        crop_type: identified crop
        confidence: 0-100
        evidence: list of evidence points
        growth_phase: current phase
        key_events: detected events (sowing, harvest, grazing)
    """
    if not observations:
        return {"crop_type": "Unknown", "confidence": 0}

    available = [o for o in observations if o.get("available")]
    if len(available) < 5:
        return {"crop_type": "Unknown", "confidence": 10,
                "evidence": ["Insufficient SAR observations"]}

    monthly = extract_monthly_vh(available)

    # Seasonal VH values
    winter_vh = np.mean([monthly[m]["vh"] for m in [12, 1, 2]
                         if m in monthly]) if any(m in monthly for m in [12, 1, 2]) else None
    spring_vh = np.mean([monthly[m]["vh"] for m in [3, 4, 5]
                         if m in monthly]) if any(m in monthly for m in [3, 4, 5]) else None
    summer_vh = np.mean([monthly[m]["vh"] for m in [6, 7, 8]
                         if m in monthly]) if any(m in monthly for m in [6, 7, 8]) else None

    # VV range — key grassland discriminator
    all_vv = [o.get("vv") or o.get("vv_mean") for o in available
              if o.get("vv") or o.get("vv_mean")]
    vv_range = round(max(all_vv) - min(all_vv), 2) if len(all_vv) > 3 else None

    # All VH values for trend analysis
    all_vh = [(o["date"], o.get("vh") or o.get("vh_mean"))
              for o in available if o.get("vh") or o.get("vh_mean")]
    all_vh.sort(key=lambda x: x[0])

    evidence = []
    events = []
    crop_scores = {
        "Grassland": 0,
        "Spring Barley": 0,
        "Winter Wheat": 0,
        "Oilseed Rape": 0,
        "Oats": 0
    }

    # === KEY DISCRIMINATORS from 479 Irish DAFM parcels ===
    # Reference signatures:
    # Permanent Pasture: amp +0.15, VV range  71, winter 28.6
    # Spring Barley:     amp +5.57, VV range 227, winter 28.6
    # Oilseed Rape:      amp -2.84, VV range 214, winter 32.6
    # Winter Wheat:      amp -9.74, VV range 190, winter 29.8
    # Oats:              amp +11.5, VV range 144, winter 41.3
    # Spring Wheat:      amp -15.5, VV range 131, winter 29.8

    # === DERIVED FEATURES — monthly resolution ===
    vh_by_month = {}
    vv_by_month = {}
    if monthly:
        vh_by_month = {m: v["vh"] for m, v in monthly.items() if v.get("vh")}
        vv_by_month = {m: v["vv"] for m, v in monthly.items() if v.get("vv")}

    # Seasonal averages
    def avg(months):
        vals = [vh_by_month[m] for m in months if m in vh_by_month]
        return round(sum(vals)/len(vals), 2) if vals else None

    winter_vh = avg([12, 1, 2])
    spring_vh = avg([3, 4, 5])
    summer_vh = avg([6, 7, 8])
    autumn_vh = avg([9, 10, 11])

    # All VV for range
    all_vv_vals = [vv_by_month[m] for m in vv_by_month]

    # Amplitude features
    vh_amplitude = round(summer_vh - winter_vh, 2) if summer_vh and winter_vh else None
    spring_minus_winter = round(spring_vh - winter_vh, 2) if spring_vh and winter_vh else None
    summer_minus_spring = round(summer_vh - spring_vh, 2) if summer_vh and spring_vh else None

    # Peak and trough months
    peak_month = max(vh_by_month, key=vh_by_month.get) if vh_by_month else None
    trough_month = min(vh_by_month, key=vh_by_month.get) if vh_by_month else None
    seasonal_amplitude = round(
        vh_by_month[peak_month] - vh_by_month[trough_month], 2
    ) if peak_month and trough_month else None

    # Monthly growth rates — key cereal discriminator
    # Apr-May growth (spring cereals germinating)
    apr_may_rate = None
    if 4 in vh_by_month and 5 in vh_by_month:
        apr_may_rate = round(vh_by_month[5] - vh_by_month[4], 2)

    # May-Jun rate (heading period)
    may_jun_rate = None
    if 5 in vh_by_month and 6 in vh_by_month:
        may_jun_rate = round(vh_by_month[6] - vh_by_month[5], 2)

    # Jun-Jul decline (harvest senescence)
    jun_jul_rate = None
    if 6 in vh_by_month and 7 in vh_by_month:
        jun_jul_rate = round(vh_by_month[7] - vh_by_month[6], 2)

    # Reference monthly signatures from 479 Irish parcels:
    # Grassland:     peak stable all year, amplitude < 10
    # Spring Barley: trough Jan-Mar, peak Jun-Jul, amplitude +5.57
    # Winter Wheat:  peak Apr-May, drops Jun-Jul, amplitude -9.74
    # Oilseed Rape:  dip Mar-Apr (flowering), peak Nov-Jan, amplitude -2.84
    # Oats:          trough Jan-Mar, peak Jul-Aug, highest amplitude +11.55
    # Spring Wheat:  sharp Jul-Aug decline, amplitude -15.51

    # Reference derived features from 479 Irish parcels:
    # Crop              amp    spring-win  seasonal_amp
    # Permanent Pasture +0.15   -4.21        varies
    # Spring Barley     +5.57   -3.13        varies
    # Oilseed Rape      -2.84   -9.48        varies
    # Winter Wheat      -9.74   -5.40        varies
    # Oats             +11.55  -11.28        varies
    # Spring Wheat     -15.51   -9.50        varies

    # RULE 1 — VV range primary discriminator
    if vv_range is not None:
        if vv_range < 100:
            crop_scores["Grassland"] += 70
            evidence.append(f"VV range {vv_range} < 100 → Permanent Pasture (ref: 71.54)")
        elif vv_range < 160:
            crop_scores["Oats"] += 30
            crop_scores["Spring Barley"] += 15
            evidence.append(f"VV range {vv_range} 100-160 → Oats/Spring Wheat range")
        elif vv_range < 200:
            crop_scores["Winter Wheat"] += 30
            crop_scores["Oilseed Rape"] += 20
            evidence.append(f"VV range {vv_range} 160-200 → Winter cereal/OSR range")
        else:
            crop_scores["Spring Barley"] += 30
            crop_scores["Oilseed Rape"] += 25
            evidence.append(f"VV range {vv_range} > 200 → Spring Barley/OSR range")

    # RULE 2 — Winter VH level
    if winter_vh is not None:
        if winter_vh > 38:
            crop_scores["Oats"] += 50
            evidence.append(f"Winter VH {round(winter_vh,1)} > 38 → Oats (ref: 41.31)")
        elif winter_vh > 31:
            crop_scores["Oilseed Rape"] += 20
            evidence.append(f"Winter VH {round(winter_vh,1)} 31-38 → OSR range (ref: 32.62)")
        elif winter_vh < 22:
            crop_scores["Grassland"] += 10
            evidence.append(f"Winter VH {round(winter_vh,1)} < 22 — low winter biomass")

    # RULE 3 — VH amplitude (summer - winter) — key cereal separator
    # References: OSR -2.84, Grassland +0.15, Barley +5.57, Oats +11.55
    #             Winter Wheat -9.74, Spring Wheat -15.51
    if vh_amplitude is not None:
        if vh_amplitude > 8:
            crop_scores["Oats"] += 50
            evidence.append(f"VH amplitude +{vh_amplitude} > 8 → Oats (ref: +11.55)")
        elif vh_amplitude > 3:
            crop_scores["Spring Barley"] += 45
            evidence.append(f"VH amplitude +{vh_amplitude} 3-8 → Spring Barley (ref: +5.57)")
        elif vh_amplitude > -2:
            crop_scores["Grassland"] += 25
            evidence.append(f"VH amplitude {vh_amplitude} near zero → Grassland (ref: +0.15)")
        elif vh_amplitude > -7:
            crop_scores["Oilseed Rape"] += 35
            evidence.append(f"VH amplitude {vh_amplitude} -2 to -7 → OSR (ref: -2.84)")
        elif vh_amplitude > -13:
            crop_scores["Winter Wheat"] += 45
            evidence.append(f"VH amplitude {vh_amplitude} -7 to -13 → Winter Wheat (ref: -9.74)")
        else:
            crop_scores["Spring Barley"] += 35
            evidence.append(f"VH amplitude {vh_amplitude} < -13 → Spring Wheat (ref: -15.51)")

    # RULE 4 — Spring minus winter (OSR flowering dip + Oats distinction)
    # References: OSR -9.48, Oats -11.28, Wheat -5.40, Barley -3.13
    if spring_minus_winter is not None:
        if spring_minus_winter < -9:
            crop_scores["Oilseed Rape"] += 35
            crop_scores["Oats"] += 20
            evidence.append(f"Spring-winter {round(spring_minus_winter,1)} < -9 → OSR flowering or Oats (ref: OSR -9.48, Oats -11.28)")
            events.append({"event": "Flowering/peak detected", "period": "March-April",
                          "signal": f"{round(spring_minus_winter,1)} VH dip"})
        elif spring_minus_winter < -4:
            crop_scores["Winter Wheat"] += 25
            evidence.append(f"Spring-winter {round(spring_minus_winter,1)} -4 to -9 → Winter Wheat (ref: -5.40)")
        elif spring_minus_winter < -1:
            crop_scores["Spring Barley"] += 15
            evidence.append(f"Spring-winter {round(spring_minus_winter,1)} → Spring Barley range (ref: -3.13)")

    # RULE 5 — Monthly growth rates (cereal subtype discriminator)
    # Peak month tells us crop type
    if peak_month is not None:
        if peak_month in [6, 7]:
            crop_scores["Oats"] += 20
            crop_scores["Spring Barley"] += 15
            evidence.append(f"Peak VH in month {peak_month} (Jun/Jul) → Spring cereal")
        elif peak_month in [4, 5]:
            crop_scores["Winter Wheat"] += 25
            evidence.append(f"Peak VH in month {peak_month} (Apr/May) → Winter cereal")
        elif peak_month in [11, 12, 1]:
            crop_scores["Oilseed Rape"] += 25
            evidence.append(f"Peak VH in month {peak_month} (Nov-Jan) → OSR established")
        elif peak_month in [2, 3, 8, 9, 10]:
            crop_scores["Grassland"] += 15
            evidence.append(f"Peak VH in month {peak_month} → Grassland pattern")

    # Trough month
    if trough_month is not None:
        if trough_month in [1, 2, 3]:
            crop_scores["Spring Barley"] += 15
            crop_scores["Oats"] += 15
            evidence.append(f"Trough VH in month {trough_month} (Jan-Mar) → Spring sown (bare soil winter)")
        elif trough_month in [7, 8]:
            crop_scores["Winter Wheat"] += 20
            crop_scores["Spring Barley"] += 10
            evidence.append(f"Trough VH in month {trough_month} (Jul/Aug) → Summer harvest")

    # Apr-May growth rate
    if apr_may_rate is not None:
        if apr_may_rate > 5:
            crop_scores["Spring Barley"] += 20
            crop_scores["Oats"] += 15
            evidence.append(f"Strong Apr-May growth +{apr_may_rate} → Spring cereal tillering")
        elif apr_may_rate < -5:
            crop_scores["Oilseed Rape"] += 20
            evidence.append(f"Apr-May decline {apr_may_rate} → OSR post-flowering")

    # Jun-Jul decline rate
    if jun_jul_rate is not None:
        if jun_jul_rate < -10:
            crop_scores["Winter Wheat"] += 25
            crop_scores["Spring Barley"] += 15
            evidence.append(f"Sharp Jun-Jul decline {jun_jul_rate} → Harvest/senescence")
        elif jun_jul_rate < -5:
            crop_scores["Winter Wheat"] += 15
            evidence.append(f"Jun-Jul decline {jun_jul_rate} → Ripening")
        elif jun_jul_rate > 3:
            crop_scores["Oats"] += 20
            evidence.append(f"Jun-Jul rise +{jun_jul_rate} → Oats late heading")

    # RULE 6 — Sudden VH drops
    vh_vals = [v for _, v in all_vh if v]
    for i in range(1, len(vh_vals)):
        drop = vh_vals[i-1] - vh_vals[i]
        if drop > 15:
            date = all_vh[i][0]
            if vv_range and vv_range < 100:
                events.append({"event": "Grazing event", "date": date,
                              "signal": f"-{round(drop,1)} VH"})
                evidence.append(f"VH drop {round(drop,1)} on {date} → grazing")
            else:
                events.append({"event": "Harvest detected", "date": date,
                              "signal": f"-{round(drop,1)} VH"})
                evidence.append(f"VH drop {round(drop,1)} on {date} → harvest")

    # === DETERMINE CROP TYPE ===
    best_crop = max(crop_scores, key=crop_scores.get)
    best_score = crop_scores[best_crop]
    total = sum(crop_scores.values())

    if total > 0:
        # Classification score = actual top probability from ranking table
        # Calculated after softmax, so it matches the displayed ranking
        classification_score = 0  # will be set after confidence_table is built
    else:
        classification_score = 20

    # Build ranked confidence table using softmax-style probabilities
    # Add small prior to all crops to avoid 0%
    prior = 2
    ranked = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)
    total_score = max(sum(v + prior for v in crop_scores.values()), 1)
    confidence_table = [
        {
            "crop": crop,
            "score": score,
            "pct": round((score + prior) / total_score * 100)
        }
        for crop, score in ranked
    ]

    # Cap confidence based on feature availability
    n_features = sum([
        vv_range is not None,
        winter_vh is not None,
        spring_vh is not None,
        summer_vh is not None,
        len(all_vh) >= 10,
        len(events) > 0
    ])
    # Max confidence scales with feature count
    top_pct = confidence_table[0]["pct"] if confidence_table else 0
    classification_score = top_pct  # matches ranking display exactly
    max_confidence = {6: 92, 5: 85, 4: 78, 3: 68, 2: 55, 1: 40, 0: 20}
    feature_cap = max_confidence.get(n_features, 55)
    operational_confidence = min(feature_cap, top_pct)

    # Detect uncertainty — top two crops within 15%
    second_pct = confidence_table[1]["pct"] if len(confidence_table) > 1 else 0
    uncertain = (top_pct - second_pct) < 15
    confidence = operational_confidence  # keep for backward compat
    if uncertain:
        uncertainty_note = (
            f"Classification uncertain — {confidence_table[0]['crop']} "
            f"({top_pct}%) vs {confidence_table[1]['crop']} "
            f"({second_pct}%) — farmer confirmation recommended"
        )
    else:
        uncertainty_note = None

    # Feature sufficiency note
    feature_note = (
        f"Based on {n_features}/6 features: "
        f"{'VV range ' if vv_range else ''}"
        f"{'Winter VH ' if winter_vh else ''}"
        f"{'Spring VH ' if spring_vh else ''}"
        f"{'Summer VH ' if summer_vh else ''}"
        f"{'Time series ' if len(all_vh) >= 10 else ''}"
        f"{'Events ' if events else ''}"
    ).strip()

    # Determine current growth phase
    current_month = datetime.now().month
    growth_phase = determine_growth_phase(best_crop, current_month,
                                          monthly, spring_vh, summer_vh)

    return {
        "crop_type": best_crop if not uncertain else "Uncertain",
        "top_crop": best_crop,
        "classification_score": classification_score,
        "confidence": operational_confidence,
        "confidence_table": confidence_table,
        "uncertain": uncertain,
        "uncertainty_note": uncertainty_note,
        "evidence": evidence,
        "growth_phase": growth_phase,
        "key_events": events,
        "seasonal_vh": {
            "winter": round(float(winter_vh), 2) if winter_vh else None,
            "spring": round(float(spring_vh), 2) if spring_vh else None,
            "summer": round(float(summer_vh), 2) if summer_vh else None
        },
        "vv_range": vv_range,
        "method": "SAR time series analysis",
        "feature_note": feature_note,
        "n_features": n_features
    }


def determine_growth_phase(crop, month, monthly, spring_vh, summer_vh):
    """Determine current growth phase from crop and month"""
    if crop == "Grassland":
        # Use VH trend to refine grassland stage
        if month in [2, 3]:
            return "Spring Recovery — growth restarting"
        elif month in [4, 5]:
            return "Rapid Growth — high demand period"
        elif month == 6:
            # Check if VH declining (post-grazing) or stable
            if summer_vh and spring_vh and summer_vh < spring_vh - 3:
                return "Post-Grazing Recovery"
            return "Peak Growth — graze or cut decision"
        elif month in [7, 8]:
            if summer_vh and summer_vh < 15:
                return "Post-Grazing Recovery"
            return "Summer Growth — monitor cover"
        elif month in [9, 10]:
            return "Autumn Flush — reducing growth"
        elif month == 11:
            return "Late Season — prepare for housing"
        else:
            return "Winter — very slow growth"

    elif crop == "Spring Barley":
        if month in [3, 4]:
            return "Sowing / Germination"
        elif month in [5, 6]:
            return "Stem Extension / Tillering"
        elif month == 7:
            return "Heading / Ear Emergence"
        elif month == 8:
            return "Ripening / Harvest"
        else:
            return "Bare soil / Stubble"

    elif crop == "Winter Wheat":
        if month in [10, 11]:
            return "Establishment / Tillering"
        elif month in [12, 1, 2]:
            return "Vernalisation / Dormancy"
        elif month in [3, 4]:
            return "Stem Extension"
        elif month in [5, 6]:
            return "Heading / Ear Emergence"
        elif month == 7:
            return "Ripening / Harvest"
        else:
            return "Bare soil / Post-harvest"

    elif crop == "Oilseed Rape":
        if month in [9, 10]:
            return "Establishment"
        elif month in [11, 12, 1, 2]:
            return "Rosette / Overwintering"
        elif month in [3, 4]:
            return "Flowering"
        elif month in [5, 6]:
            return "Pod Fill"
        elif month == 7:
            return "Harvest"
        else:
            return "Bare soil"

    elif crop == "Oats":
        if month in [3, 4]:
            return "Sowing / Germination"
        elif month in [5, 6]:
            return "Tillering / Stem Extension"
        elif month == 7:
            return "Heading"
        elif month == 8:
            return "Harvest"
        else:
            return "Bare soil / Stubble"

    return "Growing"


if __name__ == "__main__":
    import json
    import os
    import sys
    sys.path.insert(0, '/workspaces/crop-trajectory')

    os.environ['CDSE_CLIENT_ID'] = 'sh-6e5978f5-f5d6-43d6-874d-720d84121683'
    os.environ['CDSE_CLIENT_SECRET'] = 'yrMEXQ5drlF26yrB4sTEXfWOIwKtB1fP'

    print("Testing SAR Time Series Crop Identifier")
    print("=" * 50)

    with open('/workspaces/crop-trajectory/sar_ireland_2026.json') as f:
        obs = json.load(f)

    result = analyse_time_series(obs)
    print(f"Crop type:           {result['crop_type']}")
    print(f"Classification Score: {result['classification_score']}%")
    print(f"Operational Conf:     {result['confidence']}%")
    print(f"Uncertain:            {result['uncertain']}")
    print(f"Growth phase:         {result['growth_phase']}")
    print(f"\nClassification Rankings:")
    for r in result['confidence_table']:
        bar = "█" * (r['pct'] // 5)
        print(f"  {r['crop']:<20} {bar:<20} {r['pct']}%")
    if result['uncertainty_note']:
        print(f"\n⚠ {result['uncertainty_note']}")
    print(f"\nFeature sufficiency: {result.get('feature_note')}")
    print(f"Features used: {result.get('n_features')}/6")
    print(f"\nEvidence:")
    for e in result['evidence']:
        print(f"  → {e}")
    print(f"\nKey events:")
    for e in result['key_events']:
        print(f"  → {e}")
    print(f"\nSeasonal VH: {result['seasonal_vh']}")
    print(f"VV range:    {result['vv_range']}")
