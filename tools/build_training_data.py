"""
Build Irish Crop SAR Training Dataset
Uses DAFM parcels as ground truth labels
"""

import requests
import json
import time
import os
import sys
sys.path.insert(0, '/workspaces/crop-trajectory')

SAMPLE_LOCATIONS = [
    # Co. Meath — mixed arable + grassland
    (53.655, -6.675),
    (53.640, -6.650),
    (53.670, -6.700),
    (53.620, -6.620),
    # Co. Wexford — tillage area
    (52.450, -6.500),
    (52.400, -6.450),
    (52.500, -6.550),
    (52.350, -6.400),
    # Co. Kildare — mixed
    (53.200, -6.900),
    (53.150, -6.850),
    (53.250, -6.950),
    # Co. Tipperary
    (52.650, -7.800),
    (52.700, -7.750),
    # Co. Cork — tillage
    (51.900, -8.500),
    (51.850, -8.450),
]

DAFM_API = "https://cube-earth.onrender.com/parcels_in_bbox"


def get_parcels_at_location(lat, lng, bbox_km=2.0):
    delta = bbox_km / 111.0
    try:
        r = requests.get(
            DAFM_API,
            params={
                "minlng": lng - delta,
                "minlat": lat - delta,
                "maxlng": lng + delta,
                "maxlat": lat + delta
            }, timeout=30)
        if r.status_code == 200:
            return r.json().get("features", [])
    except Exception as e:
        print(f"  Error: {e}")
    return []


def collect_training_parcels():
    training_data = {}
    seen_parcels = set()

    print("Collecting DAFM parcels across Ireland...")
    print("="*50)

    for lat, lng in SAMPLE_LOCATIONS:
        print(f"Searching ({lat}, {lng})...")
        parcels = get_parcels_at_location(lat, lng)
        print(f"  Found {len(parcels)} parcels")

        for parcel in parcels:
            props = parcel.get("properties", {})
            par_lab = props.get("PAR_LAB", "")
            crop = props.get("CROP", "Unknown")
            area = props.get("CLAIM_AREA", 0)

            if par_lab in seen_parcels:
                continue
            if area < 1.0:
                continue
            if not crop or crop == "Unknown":
                continue

            seen_parcels.add(par_lab)

            if crop not in training_data:
                training_data[crop] = []

            training_data[crop].append({
                "par_lab": par_lab,
                "crop": crop,
                "area_ha": area,
                "centroid_lat": lat,
                "centroid_lng": lng,
                "polygon": parcel["geometry"]["coordinates"][0]
            })

        time.sleep(0.5)

    return training_data


if __name__ == "__main__":
    os.makedirs('/workspaces/crop-trajectory/data', exist_ok=True)
    training_data = collect_training_parcels()

    print("\nTRAINING DATA SUMMARY")
    print("="*50)
    total = 0
    for crop, parcels in sorted(training_data.items()):
        print(f"  {crop}: {len(parcels)} parcels")
        total += len(parcels)
    print(f"  Total: {total} parcels")

    output = '/workspaces/crop-trajectory/data/irish_training_parcels.json'
    with open(output, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"\nSaved to {output}")
