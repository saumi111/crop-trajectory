"""
Scene-first S2 extraction.
For each dekad → for each tile → read all parcels on tile in one pass.
"""
import requests, rasterio, pyproj, numpy as np, time, json, os
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from collections import defaultdict

BAND_MAP = [
    ('B01','coastal'), ('B02','blue'),    ('B03','green'), ('B04','red'),
    ('B05','rededge1'),('B06','rededge2'),('B07','rededge3'),('B08','nir'),
    ('B8A','nir08'),   ('B09','nir09'),   ('B10',None),    ('B11','swir16'),
    ('B12','swir22')
]
TARGET_STEPS = 32
IRELAND_BBOX = [-10.5, 51.4, -6.0, 55.4]

CROP_MAP = {
    "Wheat - Winter":"Wheat","Wheat - Spring":"Wheat",
    "Barley - Spring":"Barley","Barley - Winter":"Barley",
    "Oats - Spring":"Oats","Oats - Winter":"Oats",
    "Oilseed Rape - Winter":"Oilseed Rape",
    "Maize":"Maize","Beans - Spring":"Beans",
    "Permanent Pasture":"Grassland",
}

def dekad_dates(start_date, end_date):
    """Generate dekad start dates between start and end."""
    dates = []
    d = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        # Next dekad
        if d.day < 10: d = d.replace(day=11)
        elif d.day < 20: d = d.replace(day=21)
        else:
            if d.month == 12: d = datetime(d.year+1,1,1)
            else: d = datetime(d.year,d.month+1,1)
    return dates

def read_band_all_centroids(args):
    asset_key, assets, centroids_list = args
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

def main():
    with open("/workspaces/crop-trajectory/data/dafm_arable_parcels.json") as f:
        all_parcels = json.load(f)

    # Build parcel index
    parcels = []
    for p in all_parcels:
        crop = CROP_MAP.get(p["properties"].get("CROP",""))
        if not crop or crop=="Grassland": continue
        coords = p["geometry"]["coordinates"][0]
        lngs=[c[0] for c in coords]; lats=[c[1] for c in coords]
        parcels.append({
            "par_lab": p["properties"]["PAR_LAB"],
            "label": crop,
            "lat": (min(lats)+max(lats))/2,
            "lng": (min(lngs)+max(lngs))/2,
        })
    print(f"Total arable parcels: {len(parcels)}")

    # Storage: par_lab → list of band vectors per dekad
    ts_store = defaultdict(list)
    meta = {p["par_lab"]:p["label"] for p in parcels}

    # Get all dekads
    dekad_starts = dekad_dates("2024-10-01","2025-09-01")
    print(f"Dekads to process: {len(dekad_starts)}")

    for di, dk_start in enumerate(dekad_starts):
        dk_date = datetime.strptime(dk_start,"%Y-%m-%d")
        dk_end = (dk_date+timedelta(days=10)).strftime("%Y-%m-%d")

        # Get all scenes for this dekad across Ireland
        r = requests.post("https://earth-search.aws.element84.com/v1/search",
            json={"collections":["sentinel-2-l2a"],"bbox":IRELAND_BBOX,
                  "datetime":f"{dk_start}T00:00:00Z/{dk_end}T23:59:59Z",
                  "query":{"eo:cloud_cover":{"lt":80}},"limit":50},timeout=15)
        if r.status_code!=200: continue
        scenes = r.json().get("features",[])
        if not scenes:
            print(f"  [{di+1}/{len(dekad_starts)}] {dk_start}: no scenes")
            continue

        # Best scene per tile
        by_tile = {}
        for s in scenes:
            tile = s["id"].split("_")[1]
            cloud = s["properties"].get("eo:cloud_cover",100)
            if tile not in by_tile or cloud < by_tile[tile][0]:
                by_tile[tile] = (cloud, s)

        t0 = time.time()
        n_extracted = 0

        for tile,(cloud,scene) in by_tile.items():
            sb = scene["bbox"]
            # Parcels on this tile
            tile_parcels = [p for p in parcels
                           if sb[0]<=p["lng"]<=sb[2] and sb[1]<=p["lat"]<=sb[3]]
            if not tile_parcels: continue

            centroids = [(p["lat"],p["lng"]) for p in tile_parcels]
            assets = scene["assets"]
            band_keys = [ak for _,ak in BAND_MAP]

            with ThreadPoolExecutor(max_workers=8) as ex:
                band_results = list(ex.map(
                    lambda ak: read_band_all_centroids((ak, assets, centroids)),
                    band_keys))

            # Store per parcel
            for j,p in enumerate(tile_parcels):
                bands = [band_results[b][j] for b in range(len(BAND_MAP))]
                ts_store[p["par_lab"]].append(bands)
            n_extracted += len(tile_parcels)

        print(f"  [{di+1}/{len(dekad_starts)}] {dk_start}: "
              f"{len(by_tile)} tiles {n_extracted} parcels {time.time()-t0:.0f}s")

    # Build fixed tensors and save
    results = []
    for par_lab, obs in ts_store.items():
        if not obs: continue
        while len(obs) < TARGET_STEPS: obs.append(obs[-1].copy())
        tensor = np.array(obs[:TARGET_STEPS], dtype=np.float32)
        results.append({
            "par_lab": par_lab,
            "label": meta[par_lab],
            "timeseries": tensor.tolist()
        })

    out = "/workspaces/crop-trajectory/data/irish_s2_timeseries.json"
    with open(out,"w") as f:
        json.dump(results, f)
    print(f"\nSaved {len(results)} parcels to {out}")

if __name__=="__main__":
    main()
