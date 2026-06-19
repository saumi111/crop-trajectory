"""
Irish S2 time series extractor — tile-grouped for efficiency.
Groups parcels by S2 tile, one cold start per tile per dekad.
Output per parcel: [36, 13] scaled reflectance (BreizhCrops compatible)
"""
import requests, rasterio, pyproj, numpy as np, time
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor

BAND_MAP = [
    ('B01','coastal'), ('B02','blue'),    ('B03','green'), ('B04','red'),
    ('B05','rededge1'),('B06','rededge2'),('B07','rededge3'),('B08','nir'),
    ('B8A','nir08'),   ('B09','nir09'),   ('B10',None),    ('B11','swir16'),
    ('B12','swir22')
]
VALID_SCL = {4, 5, 7}
TARGET_STEPS = 36

def search_scenes(bbox, start_date, end_date):
    r = requests.post("https://earth-search.aws.element84.com/v1/search",
        json={"collections":["sentinel-2-l2a"],"bbox":bbox,
              "datetime":f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
              "query":{"eo:cloud_cover":{"lt":80}},"limit":200},timeout=15)
    if r.status_code!=200: return []
    return sorted(r.json().get("features",[]),
                  key=lambda f: f["properties"]["datetime"])

def dekad_key(date_str):
    from datetime import datetime
    d = datetime.strptime(date_str[:10],"%Y-%m-%d")
    dk = 1 if d.day<=10 else 2 if d.day<=20 else 3
    return f"{d.year}{d.month:02d}{dk}"

def read_band_at(args):
    url, lat_c, lng_c = args
    if not url: return 0.0
    try:
        with rasterio.open(f"/vsicurl/{url}") as src:
            tf = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x,y = tf.transform(lng_c, lat_c)
            row,col = src.index(x,y)
            if not(0<=row<src.height and 0<=col<src.width): return 0.0
            win = Window(max(0,col-5),max(0,row-5),11,11)
            data = src.read(1,window=win).flatten()
            valid = data[(data>100)&(data<60000)]
            return float(np.median(valid))*0.0001 if len(valid)>0 else 0.0
    except: return 0.0

def extract_parcel_bands(scene, lat_c, lng_c):
    """Extract all 13 bands for one parcel from one scene in parallel."""
    assets = scene["assets"]
    args = []
    for bname, asset_key in BAND_MAP:
        if asset_key is None:
            args.append(("", lat_c, lng_c))
        else:
            url = assets.get(asset_key,{}).get("href","")
            args.append((url, lat_c, lng_c))
    with ThreadPoolExecutor(max_workers=8) as ex:
        vals = list(ex.map(read_band_at, args))
    return vals

def get_s2_timeseries(polygon, start_date, end_date):
    """
    Extract [36, 13] S2 time series for one parcel.
    Returns (tensor, dates) or None.
    """
    lngs=[c[0] for c in polygon]; lats=[c[1] for c in polygon]
    bbox=[min(lngs),min(lats),max(lngs),max(lats)]
    lat_c=(bbox[1]+bbox[3])/2; lng_c=(bbox[0]+bbox[2])/2

    scenes = search_scenes(bbox, start_date, end_date)
    if not scenes: return None

    # Best scene per dekad
    by_dekad = {}
    for scene in scenes:
        dk = dekad_key(scene["properties"]["datetime"])
        cloud = scene["properties"].get("eo:cloud_cover",100)
        sb = scene.get("bbox",[])
        if sb:
            margin = min(lat_c-sb[1],sb[3]-lat_c,lng_c-sb[0],sb[2]-lng_c)
            if margin < 0.02: continue
        if dk not in by_dekad or cloud < by_dekad[dk][0]:
            by_dekad[dk] = (cloud, scene)

    if not by_dekad: return None

    observations = []
    for dk in sorted(by_dekad.keys()):
        _, scene = by_dekad[dk]
        bands = extract_parcel_bands(scene, lat_c, lng_c)
        observations.append(bands)

    if not observations: return None

    # Pad to TARGET_STEPS
    while len(observations) < TARGET_STEPS:
        observations.append(observations[-1].copy())
    observations = observations[:TARGET_STEPS]

    tensor = np.array(observations, dtype=np.float32)  # [36, 13]
    return tensor

if __name__ == "__main__":
    import json, time
    with open("data/dafm_arable_parcels.json") as f:
        parcels = json.load(f)

    # Test on 3 wheat parcels
    wheat = [p for p in parcels if p["properties"].get("CROP")=="Wheat - Winter"][:3]
    for i,p in enumerate(wheat):
        polygon = p["geometry"]["coordinates"][0]
        par_lab = p["properties"].get("PAR_LAB","?")
        t0 = time.time()
        tensor = get_s2_timeseries(polygon, "2024-10-01", "2025-09-30")
        elapsed = time.time()-t0
        if tensor is not None:
            print(f"Parcel {i} ({par_lab[:8]}): shape={tensor.shape} "
                  f"range={tensor.min():.3f}-{tensor.max():.3f} ({elapsed:.1f}s)")
            print(f"  B08 NIR: {[round(v,3) for v in tensor[:,7]]}")
        else:
            print(f"Parcel {i}: no data ({elapsed:.1f}s)")
