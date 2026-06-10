"""
SAR Time Series Extractor
Pulls Sentinel-1 VV, VH, RVI at regular intervals
for a given field location and date range.
Used to track crop development trajectory.
"""
import requests
import datetime
from typing import Optional

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

def get_cdse_token(client_id: str, client_secret: str) -> Optional[str]:
    try:
        r = requests.post(CDSE_TOKEN_URL, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }, timeout=10)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        print(f"Token error: {e}")
        return None

def get_sar_value(lat: float, lng: float, date: str, token: str) -> dict:
    """
    Get single SAR observation for a location and date
    Returns VV, VH, RVI values
    """
    evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["VV", "VH"],
      units: "LINEAR_POWER"
    }],
    output: { bands: 3 }
  };
}
function evaluatePixel(s) {
  var vv = s.VV;
  var vh = s.VH;
  var rvi = (4 * vh) / (vv + vh);
  return [vv, vh, rvi];
}
"""
    delta = 0.02
    bbox = [lng - delta, lat - delta, lng + delta, lat + delta]
    
    # Parse date and create 6-day window
    d = datetime.datetime.strptime(date, "%Y-%m-%d")
    d_from = (d - datetime.timedelta(days=3)).strftime("%Y-%m-%dT00:00:00Z")
    d_to = (d + datetime.timedelta(days=3)).strftime("%Y-%m-%dT23:59:59Z")

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": d_from, "to": d_to},
                    "acquisitionMode": "IW",
                    "polarization": "DV"
                },
                "processing": {"orthorectify": True, "backCoeff": "GAMMA0_TERRAIN"}
            }]
        },
        "evalscript": evalscript,
        "output": {
            "width": 5, "height": 5,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
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
            import numpy as np
            import tifffile, io
            arr = tifffile.imread(io.BytesIO(r.content))
            if arr is not None and arr.size > 0:
                if len(arr.shape) == 3:
                    vv_band = arr[:,:,0].flatten()
                    vh_band = arr[:,:,1].flatten()
                    rvi_band = arr[:,:,2].flatten()
                else:
                    vv_band = arr.flatten()
                    vh_band = arr.flatten()
                    rvi_band = arr.flatten()
                
                vv_vals = vv_band[vv_band > 0]
                vh_vals = vh_band[vh_band > 0]
                rvi_vals = rvi_band[(rvi_band > 0) & (rvi_band < 1)]
                
                if len(vv_vals) > 0 and len(vh_vals) > 0:
                    vv_mean = float(np.mean(vv_vals))
                    vh_mean = float(np.mean(vh_vals))
                    rvi_mean = float(np.mean(rvi_vals)) if len(rvi_vals) > 0 else (4*vh_mean)/(vv_mean+vh_mean)
                    return {
                        "date": date,
                        "vv": round(vv_mean, 4),
                        "vh": round(vh_mean, 4),
                        "rvi": round(rvi_mean, 4),
                        "available": True
                    }
        return {"date": date, "available": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"date": date, "available": False, "error": str(e)}


def get_sar_timeseries(lat: float, lng: float, 
                        start_date: str, end_date: str,
                        client_id: str, client_secret: str,
                        interval_days: int = 12) -> list:
    """
    Get SAR time series for a location between two dates
    Returns list of observations at regular intervals
    """
    token = get_cdse_token(client_id, client_secret)
    if not token:
        return []

    results = []
    current = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"Fetching SAR for {date_str}...")
        obs = get_sar_value(lat, lng, date_str, token)
        results.append(obs)
        current += datetime.timedelta(days=interval_days)

    return results


if __name__ == "__main__":
    import os, json
    
    client_id = os.environ.get("CDSE_CLIENT_ID", "")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")
    
    # Test on Lincolnshire wheat field
    lat, lng = 53.2307, -0.5406
    
    # Get 2025 wheat season — Oct 2024 to Aug 2025
    results = get_sar_timeseries(
        lat, lng,
        start_date="2024-10-01",
        end_date="2025-08-01",
        client_id=client_id,
        client_secret=client_secret,
        interval_days=12
    )
    
    print("\n=== SAR Time Series ===")
    for r in results:
        if r.get("available"):
            print(f"{r['date']}: VV={r['vv']} VH={r['vh']} RVI={r['rvi']}")
        else:
            print(f"{r['date']}: No data — {r.get('error','')}")
    
    # Save results
    with open("sar_timeseries_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to sar_timeseries_test.json")
