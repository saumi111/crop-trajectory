"""
Irish S2 time series extractor — tile-grouped batch mode.
One file open per band per scene serves ALL parcels on that tile.
Output: {par_lab: np.array([T, 13])} for all parcels in batch.
"""
import requests, rasterio, pyproj, numpy as np, time, json
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict

BAND_MAP = [
    ('B01','coastal'), ('B02','blue'),    ('B03','green'), ('B04','red'),
    ('B05','rededge1'),('B06','rededge2'),('B07','rededge3'),('B08','nir'),
    ('B8A','nir08'),   ('B09','nir09'),   ('B10',None),    ('B11','swir16'),
    ('B12','swir22')
]
TARGET_STEPS = 32

def dekad_key(date_str):
    d = datetime.strptime(date_str[:10], "%Y-%m-%d")
    dk = 1 if d.day<=10 else 2 if d.day<=20 else 3
    return f"{d.year}{d.month:02d}{dk}"

def search_scenes(bbox, start_date, end_date, max_cloud=80):
    r = requests.post("https://earth-search.aws.element84.com/v1/search",
        json={"collections":["sentinel-2-l2a"],"bbox":bbox,
              "datetime":f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
              "query":{"eo:cloud_cover":{"lt":max_cloud}},"limit":200},timeout=15)
    if r.status_code!=200: return []
    return sorted(r.json().get("features",[]),
                  key=lambda f: f["properties"]["datetime"])

def read_band_for_centroids(asset_key, assets, centroids_list):
    """Open one band file, read all centroids in one pass."""
    if not asset_key: return [0.0]*len(centroids_list)
    url = assets.get(asset_key,{}).get("href","")
    if not url: return [0.0]*len(centroids_list)
    try:
        with rasterio.open(f"/vsicurl/{url}") as src:
            tf = pyproj.Transformer.from_crs("EPSG:4326",src.crs,always_xy=True)
            results = []
            for lat,lng in centroids_list:
                x,y = tf.transform(lng,lat)
                row,col = src.index(x,y)
                if not(0<=row<src.height and 0<=col<src.width):
                    results.append(0.0); continue
                win = Window(max(0,col-5),max(0,row-5),11,11)
                data = src.read(1,window=win).flatten()
                valid = data[(data>100)&(data<60000)]
                results.append(float(np.median(valid))*0.0001 if len(valid)>0 else 0.0)
            return results
    except: return [0.0]*len(centroids_list)

def extract_batch(parcels_with_labels, start_date, end_date):
    """
    Extract S2 time series for a batch of parcels.
    Args:
        parcels_with_labels: list of (par_lab, label, polygon)
    Returns:
        dict: {par_lab: {"label":str, "timeseries":np.array([T,13])}}
    """
    # Compute centroids
    centroids = []
    par_labs = []
    labels = []
    for par_lab, label, polygon in parcels_with_labels:
        lngs=[c[0] for c in polygon]; lats=[c[1] for c in polygon]
        centroids.append(((min(lats)+max(lats))/2,(min(lngs)+max(lngs))/2))
        par_labs.append(par_lab)
        labels.append(label)

    # Use first parcel bbox to find scenes
    lat0,lng0 = centroids[0]
    all_lats=[c[0] for c in centroids]; all_lngs=[c[1] for c in centroids]
    bbox=[min(all_lngs)-0.05,min(all_lats)-0.05,
          max(all_lngs)+0.05,max(all_lats)+0.05]

    scenes = search_scenes(bbox, start_date, end_date)
    if not scenes:
        print("  No scenes found"); return {}

    # Best scene per dekad
    by_dekad = {}
    for s in scenes:
        key = dekad_key(s["properties"]["datetime"])
        cloud = s["properties"].get("eo:cloud_cover",100)
        sb = s.get("bbox",[])
        if sb:
            # Check at least one centroid is in this scene
            in_scene = any(sb[0]<=lng<=sb[2] and sb[1]<=lat<=sb[3]
                          for lat,lng in centroids)
            if not in_scene: continue
        if key not in by_dekad or cloud < by_dekad[key][0]:
            by_dekad[key] = (cloud, s)

    print(f"  {len(by_dekad)} dekads found")

    # Per-parcel time series storage
    ts_data = {pl: [] for pl in par_labs}

    for dk in sorted(by_dekad.keys()):
        _, scene = by_dekad[dk]
        assets = scene["assets"]

        # Find which centroids are in this scene
        sb = scene.get("bbox",[])
        active_indices = [i for i,(lat,lng) in enumerate(centroids)
                         if not sb or (sb[0]<=lng<=sb[2] and sb[1]<=lat<=sb[3])]
        active_centroids = [centroids[i] for i in active_indices]

        if not active_centroids: continue

        # Read all bands in parallel, all active parcels per band
        band_keys = [ak for _,ak in BAND_MAP]
        with ThreadPoolExecutor(max_workers=8) as ex:
            band_results = list(ex.map(
                lambda ak: read_band_for_centroids(ak, assets, active_centroids),
                band_keys))

        # band_results[band_idx][parcel_idx]
        for j, i in enumerate(active_indices):
            bands = [band_results[b][j] for b in range(len(BAND_MAP))]
            ts_data[par_labs[i]].append(bands)

    # Build fixed-length tensors
    results = {}
    for pl, label in zip(par_labs, labels):
        obs = ts_data[pl]
        if not obs: continue
        while len(obs) < TARGET_STEPS:
            obs.append(obs[-1].copy())
        tensor = np.array(obs[:TARGET_STEPS], dtype=np.float32)
        results[pl] = {"label": label, "timeseries": tensor}

    return results


if __name__ == "__main__":
    import json, time

    CROP_MAP = {
        "Wheat - Winter":"Wheat","Wheat - Spring":"Wheat",
        "Barley - Spring":"Barley","Barley - Winter":"Barley",
        "Oats - Spring":"Oats","Oats - Winter":"Oats",
        "Oilseed Rape - Winter":"Oilseed Rape",
        "Maize":"Maize","Beans - Spring":"Beans",
        "Permanent Pasture":"Grassland",
    }

    with open("data/dafm_arable_parcels.json") as f:
        all_parcels = json.load(f)

    # Pilot: 5 wheat parcels
    pilot = []
    for p in all_parcels:
        crop = CROP_MAP.get(p["properties"].get("CROP",""))
        if crop == "Wheat" and len(pilot) < 5:
            pilot.append((
                p["properties"]["PAR_LAB"],
                crop,
                p["geometry"]["coordinates"][0]
            ))

    print(f"Extracting {len(pilot)} pilot parcels...")
    t0 = time.time()
    results = extract_batch(pilot, "2024-10-01", "2025-09-30")
    elapsed = time.time()-t0

    print(f"\nDone in {elapsed:.0f}s ({elapsed/len(results):.0f}s/parcel)")
    for pl, data in results.items():
        ts = data["timeseries"]
        print(f"  {pl[:12]} {data['label']}: shape={ts.shape} "
              f"B08={[round(v,3) for v in ts[:,7][:8]]}...")
