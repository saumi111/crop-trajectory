"""
SAR Polygon Extractor
Extracts SAR statistics within a field polygon
Rather than a point average

Input: GeoJSON polygon coordinates
Output: VV, VH, RVI statistics for pixels within boundary

This gives true field-level SAR signal
not a 2km area average around a point
"""

import requests
import numpy as np
from datetime import datetime, timedelta
import os


CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


def polygon_to_bbox(polygon_coords):
    """
    Get bounding box from polygon coordinates
    polygon_coords: list of [lng, lat] pairs
    Returns: [min_lng, min_lat, max_lng, max_lat]
    """
    lngs = [p[0] for p in polygon_coords]
    lats = [p[1] for p in polygon_coords]
    return [min(lngs), min(lats), max(lngs), max(lats)]


def get_sar_polygon(polygon_coords, date, token,
                    resolution_m=10):
    """
    Get SAR statistics within a field polygon
    
    Args:
        polygon_coords: list of [lng, lat] pairs
        date: date string YYYY-MM-DD
        token: CDSE access token
        resolution_m: pixel resolution in metres
    
    Returns:
        dict with VV, VH, RVI mean and std
    """
    d = datetime.strptime(date, "%Y-%m-%d")
    d_from = (d - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00Z")
    d_to = (d + timedelta(days=3)).strftime("%Y-%m-%dT23:59:59Z")

    bbox = polygon_to_bbox(polygon_coords)

    evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["VV", "VH"],
      units: "LINEAR_POWER"
    }],
    output: {bands: 3, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) {
  var vv = s.VV * 1000;
  var vh = s.VH * 1000;
  var rvi = (4 * vh) / (vv + vh + 0.0001);
  return [vv, vh, rvi];
}
"""

    # Calculate output dimensions from bbox and resolution
    lng_range = bbox[2] - bbox[0]
    lat_range = bbox[3] - bbox[1]
    # Approximate pixels needed
    width = max(10, min(100, int(lng_range * 111000 / resolution_m)))
    height = max(10, min(100, int(lat_range * 111000 / resolution_m)))

    payload = {
        "input": {
            "bounds": {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coords]
                },
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                }
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": d_from, "to": d_to},
                    "acquisitionMode": "IW",
                    "polarization": "DV"
                },
                "processing": {
                    "orthorectify": True,
                    "backCoeff": "GAMMA0_TERRAIN"
                }
            }]
        },
        "evalscript": evalscript,
        "output": {
            "width": width,
            "height": height,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }]
        }
    }

    try:
        r = requests.post(
            CDSE_PROCESS_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        if r.status_code == 200:
            import tifffile, io
            arr = tifffile.imread(io.BytesIO(r.content))

            if arr is not None and arr.size > 0:
                if len(arr.shape) == 3:
                    vv_band = arr[:,:,0].flatten()
                    vh_band = arr[:,:,1].flatten()
                    rvi_band = arr[:,:,2].flatten()
                else:
                    return {"date": date, "available": False}

                vv_valid = vv_band[vv_band > 0]
                vh_valid = vh_band[vh_band > 0]
                rvi_valid = rvi_band[(rvi_band > 0) & (rvi_band < 1)]

                if len(vv_valid) > 0:
                    return {
                        "date": date,
                        "available": True,
                        "pixels": len(vv_valid),
                        "vv_mean": round(float(np.mean(vv_valid)), 4),
                        "vv_std": round(float(np.std(vv_valid)), 4),
                        "vh_mean": round(float(np.mean(vh_valid)), 4),
                        "vh_std": round(float(np.std(vh_valid)), 4),
                        "rvi_mean": round(float(np.mean(rvi_valid)), 4),
                        "rvi_std": round(float(np.std(rvi_valid)), 4),
                        # Field variability — high std = variable crop
                        "field_variability": round(
                            float(np.std(rvi_valid) / np.mean(rvi_valid))
                            if np.mean(rvi_valid) > 0 else 0, 4)
                    }

        return {"date": date, "available": False,
                "error": f"HTTP {r.status_code}"}

    except Exception as e:
        return {"date": date, "available": False, "error": str(e)}


def get_sar_timeseries_polygon(polygon_coords,
                                start_date, end_date,
                                client_id, client_secret,
                                interval_days=12):
    """
    Get SAR time series for a field polygon
    Returns field-level statistics at each date
    """
    from extractors.sar_timeseries import get_cdse_token

    token = get_cdse_token(client_id, client_secret)
    if not token:
        return []

    results = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"Fetching SAR polygon {date_str}...")
        obs = get_sar_polygon(polygon_coords, date_str, token)
        # Rename for compatibility with point extractor
        if obs.get("available"):
            obs["vv"] = obs.get("vv_mean")
            obs["vh"] = obs.get("vh_mean")
            obs["rvi"] = obs.get("rvi_mean")
        results.append(obs)
        current += timedelta(days=interval_days)

    return results


if __name__ == "__main__":
    import os, sys, json
    sys.path.insert(0, '/workspaces/crop-trajectory')

    os.environ['CDSE_CLIENT_ID'] = \
        'sh-6e5978f5-f5d6-43d6-874d-720d84121683'
    os.environ['CDSE_CLIENT_SECRET'] = \
        'yrMEXQ5drlF26yrB4sTEXfWOIwKtB1fP'

    # Test with a real Irish field polygon
    # Small arable field in Co. Meath
    test_polygon = [
        [-6.68, 53.66],
        [-6.67, 53.66],
        [-6.67, 53.65],
        [-6.68, 53.65],
        [-6.68, 53.66]
    ]

    print("Testing SAR Polygon Extractor — Ireland")
    print(f"Field polygon: {len(test_polygon)} vertices")
    bbox = polygon_to_bbox(test_polygon)
    print(f"Bounding box: {bbox}")

    results = get_sar_timeseries_polygon(
        test_polygon,
        "2026-04-01",
        "2026-06-12",
        os.environ['CDSE_CLIENT_ID'],
        os.environ['CDSE_CLIENT_SECRET'],
        interval_days=12
    )

    available = [r for r in results if r.get("available")]
    print(f"\nGot {len(available)} observations")
    print("\nDate        VV_mean  VH_mean  RVI     Pixels  Variability")
    for r in available:
        print(f"{r['date']}  {r['vv_mean']:6.2f}   "
              f"{r['vh_mean']:5.2f}    {r['rvi_mean']:.4f}  "
              f"{r['pixels']:5}   {r['field_variability']:.4f}")
