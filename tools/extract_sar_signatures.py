"""
Extract SAR signatures for each Irish crop type
Uses real DAFM parcels as ground truth
Builds crop-specific SAR statistics for classifier improvement
"""

import json
import os
import sys
import time
import numpy as np
sys.path.insert(0, '/workspaces/crop-trajectory')

os.environ['CDSE_CLIENT_ID'] = 'sh-6e5978f5-f5d6-43d6-874d-720d84121683'
os.environ['CDSE_CLIENT_SECRET'] = 'yrMEXQ5drlF26yrB4sTEXfWOIwKtB1fP'

from extractors.sar_polygon import get_sar_timeseries_polygon

# Target crops — enough samples for each
TARGET_CROPS = {
    "Permanent Pasture": 5,
    "Barley - Spring": 5,
    "Oilseed Rape - Winter": 5,
    "Wheat - Winter": 5,
    "Oats - Spring": 5,
    "Maize": 3,
    "Wheat - Spring": 3,
}

def extract_seasonal_stats(obs):
    """Extract seasonal statistics from SAR observations"""
    if not obs:
        return None

    monthly_vv = {}
    monthly_vh = {}

    for o in obs:
        if not o.get("available"):
            continue
        month = int(o["date"].split("-")[1])
        if month not in monthly_vv:
            monthly_vv[month] = []
            monthly_vh[month] = []
        if o.get("vv"):
            monthly_vv[month].append(o["vv"])
        if o.get("vh"):
            monthly_vh[month].append(o["vh"])

    avg_vv = {m: round(np.mean(v), 2)
              for m, v in monthly_vv.items() if v}
    avg_vh = {m: round(np.mean(v), 2)
              for m, v in monthly_vh.items() if v}

    all_vv = [o["vv"] for o in obs if o.get("available") and o.get("vv")]
    all_vh = [o["vh"] for o in obs if o.get("available") and o.get("vh")]
    all_rvi = [o["rvi"] for o in obs if o.get("available") and o.get("rvi")]

    # Seasonal groups
    winter_vh = np.mean([avg_vh.get(m, 0) for m in [12, 1, 2] if m in avg_vh]) if avg_vh else 0
    spring_vh = np.mean([avg_vh.get(m, 0) for m in [3, 4, 5] if m in avg_vh]) if avg_vh else 0
    summer_vh = np.mean([avg_vh.get(m, 0) for m in [6, 7, 8] if m in avg_vh]) if avg_vh else 0
    autumn_vh = np.mean([avg_vh.get(m, 0) for m in [9, 10, 11] if m in avg_vh]) if avg_vh else 0

    return {
        "mean_vv": round(np.mean(all_vv), 2) if all_vv else None,
        "mean_vh": round(np.mean(all_vh), 2) if all_vh else None,
        "mean_rvi": round(np.mean(all_rvi), 2) if all_rvi else None,
        "std_vv": round(np.std(all_vv), 2) if all_vv else None,
        "vv_range": round(max(all_vv) - min(all_vv), 2) if all_vv else None,
        "winter_vh": round(float(winter_vh), 2),
        "spring_vh": round(float(spring_vh), 2),
        "summer_vh": round(float(summer_vh), 2),
        "autumn_vh": round(float(autumn_vh), 2),
        "monthly_vv": avg_vv,
        "monthly_vh": avg_vh,
        "n_obs": len(all_vv)
    }


if __name__ == "__main__":
    # Load training parcels
    with open('/workspaces/crop-trajectory/data/irish_training_parcels.json') as f:
        training_data = json.load(f)

    signatures = {}

    for crop_name, max_samples in TARGET_CROPS.items():
        parcels = training_data.get(crop_name, [])
        if not parcels:
            print(f"\nSkipping {crop_name} — no parcels")
            continue

        print(f"\n{'='*50}")
        print(f"Extracting SAR for: {crop_name}")
        print(f"Parcels available: {len(parcels)}, using: {min(max_samples, len(parcels))}")

        crop_stats = []
        for parcel in parcels[:max_samples]:
            print(f"  Parcel area: {parcel['area_ha']} ha...")
            try:
                obs = get_sar_timeseries_polygon(
                    parcel["polygon"],
                    "2025-10-01", "2026-06-12",
                    os.environ["CDSE_CLIENT_ID"],
                    os.environ["CDSE_CLIENT_SECRET"],
                    interval_days=12
                )
                available = [o for o in obs if o.get("available")]
                if available:
                    stats = extract_seasonal_stats(available)
                    if stats:
                        stats["crop"] = crop_name
                        stats["area_ha"] = parcel["area_ha"]
                        crop_stats.append(stats)
                        print(f"    VH winter:{stats['winter_vh']} spring:{stats['spring_vh']} summer:{stats['summer_vh']} RVI:{stats['mean_rvi']}")
                time.sleep(1)
            except Exception as e:
                print(f"    Error: {e}")

        if crop_stats:
            # Average across parcels
            signatures[crop_name] = {
                "n_parcels": len(crop_stats),
                "mean_vv": round(np.mean([s["mean_vv"] for s in crop_stats if s["mean_vv"]]), 2),
                "mean_vh": round(np.mean([s["mean_vh"] for s in crop_stats if s["mean_vh"]]), 2),
                "mean_rvi": round(np.mean([s["mean_rvi"] for s in crop_stats if s["mean_rvi"]]), 2),
                "std_vv": round(np.mean([s["std_vv"] for s in crop_stats if s["std_vv"]]), 2),
                "vv_range": round(np.mean([s["vv_range"] for s in crop_stats if s["vv_range"]]), 2),
                "winter_vh": round(np.mean([s["winter_vh"] for s in crop_stats]), 2),
                "spring_vh": round(np.mean([s["spring_vh"] for s in crop_stats]), 2),
                "summer_vh": round(np.mean([s["summer_vh"] for s in crop_stats]), 2),
                "autumn_vh": round(np.mean([s["autumn_vh"] for s in crop_stats]), 2),
                "individual_parcels": crop_stats
            }

    # Save signatures
    output = '/workspaces/crop-trajectory/data/irish_sar_signatures.json'
    with open(output, 'w') as f:
        json.dump(signatures, f, indent=2)

    print("\n" + "="*50)
    print("SAR SIGNATURES SUMMARY")
    print("="*50)
    for crop, sig in signatures.items():
        print(f"\n{crop} ({sig['n_parcels']} parcels):")
        print(f"  Winter VH: {sig['winter_vh']}")
        print(f"  Spring VH: {sig['spring_vh']}")
        print(f"  Summer VH: {sig['summer_vh']}")
        print(f"  Mean RVI:  {sig['mean_rvi']}")
        print(f"  VV range:  {sig['vv_range']}")

    print(f"\nSaved to {output}")
