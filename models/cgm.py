"""
Crop Growth Model (CGM)
Integrates SAR + Optical + Weather + Physiological data
Matches Origin Digital's approach:
SAR + optical satellite + weather + crop physiology

Output:
- Growth stage (BBCH)
- LAI
- Biomass  
- Canopy cover
- GDD accumulated
- Yield prediction
- Management alerts
"""

import requests
import numpy as np
from datetime import datetime, timedelta
import os
import sys
sys.path.insert(0, '/workspaces/crop-trajectory')

from models.crop_classifier import full_field_analysis, classify_crop
from models.physiological import get_current_physiology
from extractors.sar_timeseries import get_sar_timeseries, get_cdse_token


def get_ndvi_current(lat, lng, token, days_back=15):
    """Get current NDVI from Sentinel-2"""
    from datetime import datetime, timedelta
    today = datetime.now()
    d_from = (today - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00Z")
    d_to = today.strftime("%Y-%m-%dT23:59:59Z")
    delta = 0.02
    bbox = [lng-delta, lat-delta, lng+delta, lat+delta]

    evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{bands:["B04","B08","B05"],units:"REFLECTANCE"}],
    output: {bands:3, sampleType:"FLOAT32"}
  };
}
function evaluatePixel(s) {
  var ndvi = (s.B08-s.B04)/(s.B08+s.B04+0.0001);
  var ndre = (s.B08-s.B05)/(s.B08+s.B05+0.0001);
  var evi  = 2.5*(s.B08-s.B04)/(s.B08+6*s.B04-7.5*0.0002+1);
  return [ndvi, ndre, evi];
}
"""
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": d_from, "to": d_to},
                    "maxCloudCoverage": 60
                }
            }]
        },
        "evalscript": evalscript,
        "output": {
            "width": 5, "height": 5,
            "responses": [{"identifier": "default",
                          "format": {"type": "image/tiff"}}]
        }
    }

    try:
        r = requests.post(
            "https://sh.dataspace.copernicus.eu/api/v1/process",
            json=payload,
            headers={"Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"},
            timeout=30)
        if r.status_code == 200:
            import tifffile, io
            arr = tifffile.imread(io.BytesIO(r.content))
            if arr is not None and arr.size > 0:
                if len(arr.shape) == 3:
                    ndvi_vals = arr[:,:,0].flatten()
                    ndre_vals = arr[:,:,1].flatten()
                    evi_vals  = arr[:,:,2].flatten()
                else:
                    ndvi_vals = arr.flatten()
                    ndre_vals = arr.flatten()
                    evi_vals  = arr.flatten()

                ndvi_valid = ndvi_vals[(ndvi_vals > -0.1) & (ndvi_vals < 1.0)]
                ndre_valid = ndre_vals[(ndre_vals > -0.1) & (ndre_vals < 1.0)]
                evi_valid  = evi_vals[(evi_vals > -0.5) & (evi_vals < 2.0)]

                return {
                    "ndvi": round(float(np.mean(ndvi_valid)), 4)
                            if len(ndvi_valid) > 0 else None,
                    "ndre": round(float(np.mean(ndre_valid)), 4)
                            if len(ndre_valid) > 0 else None,
                    "evi":  round(float(np.mean(evi_valid)), 4)
                            if len(evi_valid) > 0 else None,
                    "cloud_free": True
                }
    except Exception as e:
        pass

    return {"ndvi": None, "ndre": None, "evi": None, "cloud_free": False}


def get_weather_cgm(lat, lng, start_date, end_date):
    """
    Get weather data for CGM
    Returns GDD, rainfall, temperature stats
    """
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lng,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ",".join([
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "et0_fao_evapotranspiration"
                ]),
                "timezone": "Europe/Dublin"
            }, timeout=20)

        if r.status_code == 200:
            data = r.json().get("daily", {})
            temps_max = data.get("temperature_2m_max", [])
            temps_min = data.get("temperature_2m_min", [])
            rain = data.get("precipitation_sum", [])
            et0 = data.get("et0_fao_evapotranspiration", [])

            # Calculate GDD base 0°C for wheat/barley
            gdd_total = 0
            for mx, mn in zip(temps_max, temps_min):
                if mx and mn:
                    avg = (mx + mn) / 2
                    gdd_total += max(0, avg)

            # Calculate GDD base 8°C for potato
            gdd_potato = 0
            for mx, mn in zip(temps_max, temps_min):
                if mx and mn:
                    avg = (mx + mn) / 2
                    gdd_potato += max(0, avg - 8)

            total_rain = sum([r for r in rain if r])
            avg_temp = np.mean([(mx+mn)/2 for mx, mn
                               in zip(temps_max, temps_min)
                               if mx and mn])
            total_et0 = sum([e for e in et0 if e])

            return {
                "gdd_base0": round(gdd_total, 1),
                "gdd_base8": round(gdd_potato, 1),
                "total_rainfall_mm": round(total_rain, 1),
                "avg_temp_c": round(float(avg_temp), 1),
                "total_et0_mm": round(total_et0, 1),
                "water_balance_mm": round(total_rain - total_et0, 1),
                "n_days": len(temps_max)
            }
    except Exception as e:
        pass

    return None


def run_cgm(lat, lng, client_id=None, client_secret=None):
    """
    Full Crop Growth Model
    Combines SAR + Optical + Weather + Physiology

    Args:
        lat, lng: field coordinates
        client_id, client_secret: CDSE credentials

    Returns:
        Complete CGM output
    """
    client_id = client_id or os.environ.get("CDSE_CLIENT_ID")
    client_secret = client_secret or os.environ.get("CDSE_CLIENT_SECRET")

    print(f"Running CGM for {lat}, {lng}")

    # Step 1 — Get SAR time series (current season)
    print("  Fetching SAR time series...")
    today = datetime.now()
    season_start = datetime(today.year - 1, 10, 1).strftime("%Y-%m-%d")
    season_end = today.strftime("%Y-%m-%d")

    sar_obs = get_sar_timeseries(
        lat, lng,
        season_start, season_end,
        client_id, client_secret,
        interval_days=12
    )
    available_obs = [o for o in sar_obs if o.get("available")]
    print(f"  Got {len(available_obs)} SAR observations")

    # Step 2 — Classify crop type
    print("  Classifying crop type...")
    classification = classify_crop(available_obs)
    crop_type = classification["crop_type"]
    conf = classification.get('confidence_pct', classification.get('confidence', 0))
    print(f"  Detected: {crop_type} ({conf}%)")

    # Step 3 — Run growth model
    print("  Running growth model...")
    field_analysis = full_field_analysis(available_obs)
    growth = field_analysis["field_analysis"]

    # Step 4 — Get optical data
    print("  Fetching optical data...")
    token = get_cdse_token(client_id, client_secret)
    optical = get_ndvi_current(lat, lng, token) if token else {}

    # Step 5 — Get weather data
    print("  Fetching weather data...")
    sowing_date = growth.get("planting_date")
    wx_start = sowing_date if sowing_date else season_start
    weather = get_weather_cgm(lat, lng, wx_start, season_end)

    # Step 6 — Get physiological parameters
    print("  Calculating physiological parameters...")
    physiology = get_current_physiology(available_obs, crop_type)

    # Step 7 — Refine LAI using NDVI if available
    ndvi = optical.get("ndvi")
    ndre = optical.get("ndre")
    refined_lai = None

    if ndvi and physiology:
        # NDVI-based LAI refinement
        # LAI = -ln(1 - fPAR) / k where fPAR ≈ 1.26 * NDVI
        fpar = min(0.95, 1.26 * ndvi)
        ndvi_lai = round(-np.log(max(0.001, 1 - fpar)) / 0.5, 2)
        sar_lai = physiology.get("current_lai", 0) or 0
        # Weighted average — SAR 40%, NDVI 60%
        refined_lai = round(0.4 * sar_lai + 0.6 * ndvi_lai, 2)

    # Step 8 — Compile complete CGM output
    result = {
        "location": {"lat": lat, "lng": lng},
        "analysis_date": today.strftime("%Y-%m-%d"),
        "crop_intelligence": {
            "crop_type": crop_type,
            "classification_confidence": classification["confidence_pct"],
            "planting_date": growth.get("planting_date"),
            "current_stage": growth.get("current_stage"),
            "yield_estimate_tha": growth.get("yield_estimate_tha"),
            "yield_range": growth.get("yield_range"),
            "management_alerts": growth.get("management_alerts", [])
        },
        "sar_data": {
            "observations": len(available_obs),
            "latest_date": available_obs[-1]["date"] if available_obs else None,
            "latest_vv": available_obs[-1].get("vv") if available_obs else None,
            "latest_vh": available_obs[-1].get("vh") if available_obs else None,
            "latest_rvi": available_obs[-1].get("rvi") if available_obs else None
        },
        "optical_data": {
            "ndvi": ndvi,
            "ndre": ndre,
            "evi": optical.get("evi"),
            "cloud_free": optical.get("cloud_free", False),
            "nitrogen_status": (
                "adequate" if ndre and ndre > 0.3 else
                "low" if ndre and ndre < 0.2 else
                "moderate") if ndre else None
        },
        "physiological": {
            "lai_sar": physiology.get("current_lai") if physiology else None,
            "lai_refined": refined_lai,
            "biomass_t_ha": physiology.get("current_biomass_t_ha") if physiology else None,
            "canopy_cover_pct": physiology.get("current_canopy_cover_pct") if physiology else None,
            "peak_lai": physiology.get("peak_lai_season") if physiology else None,
            "lai_trend": physiology.get("lai_trend") if physiology else None
        },
        "weather": {
            "gdd_accumulated_base0": weather.get("gdd_base0") if weather else None,
            "gdd_accumulated_base8": weather.get("gdd_base8") if weather else None,
            "total_rainfall_mm": weather.get("total_rainfall_mm") if weather else None,
            "avg_temp_c": weather.get("avg_temp_c") if weather else None,
            "water_balance_mm": weather.get("water_balance_mm") if weather else None
        }
    }

    return result


if __name__ == "__main__":
    # Test on Irish field
    lat, lng = 53.6529, -6.6789

    result = run_cgm(
        lat, lng,
        os.environ.get("CDSE_CLIENT_ID"),
        os.environ.get("CDSE_CLIENT_SECRET")
    )

    print("\n=== CROP GROWTH MODEL OUTPUT ===")
    ci = result["crop_intelligence"]
    print(f"\nCrop Intelligence:")
    print(f"  Crop type:        {ci['crop_type']} ({ci['classification_confidence']}%)")
    print(f"  Planting date:    {ci['planting_date']}")
    print(f"  Current stage:    {ci['current_stage']}")
    print(f"  Yield estimate:   {ci['yield_estimate_tha']} t/ha")
    print(f"  Yield range:      {ci['yield_range']}")

    opt = result["optical_data"]
    print(f"\nOptical Data:")
    print(f"  NDVI:             {opt['ndvi']}")
    print(f"  NDRE:             {opt['ndre']}")
    print(f"  Nitrogen status:  {opt['nitrogen_status']}")

    phy = result["physiological"]
    print(f"\nPhysiological:")
    print(f"  LAI (SAR):        {phy['lai_sar']}")
    print(f"  LAI (refined):    {phy['lai_refined']}")
    print(f"  Biomass:          {phy['biomass_t_ha']} t/ha DM")
    print(f"  Canopy cover:     {phy['canopy_cover_pct']}%")
    print(f"  LAI trend:        {phy['lai_trend']}")

    wx = result["weather"]
    print(f"\nWeather:")
    print(f"  GDD base 0°C:     {wx['gdd_accumulated_base0']}°C-days")
    print(f"  GDD base 8°C:     {wx['gdd_accumulated_base8']}°C-days")
    print(f"  Total rainfall:   {wx['total_rainfall_mm']} mm")
    print(f"  Water balance:    {wx['water_balance_mm']} mm")
    print(f"  Avg temperature:  {wx['avg_temp_c']}°C")

    if ci['management_alerts']:
        print(f"\nManagement Alerts:")
        for alert in ci['management_alerts']:
            print(f"  → {alert}")
