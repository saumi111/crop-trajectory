"""
CropFusion Classifier v18
Production-ready crop identification from SAR + NDVI + NDRE time series

Performance (5-fold CV on 250 Irish DAFM parcels):
  Force-classify all:     68%
  Confident only >50%:    90%  (47% of parcels)
  Confident only >60%:    92%  (30% of parcels)

Usage:
    from models.cropfusion_classifier import classify_crop
    result = classify_crop(sar_observations, monthly_ndvi, monthly_ndre)
"""

import numpy as np
import joblib
import os
from scipy import stats
from scipy.integrate import trapezoid

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained/cropfusion_v18.pkl")
LE_PATH    = os.path.join(os.path.dirname(__file__), "trained/cropfusion_le.pkl")

CONFIDENCE_THRESHOLD = 0.50  # 90% accuracy at this threshold

_pipe = None
_le   = None

def _load():
    global _pipe, _le
    if _pipe is None:
        _pipe = joblib.load(MODEL_PATH)
        _le   = joblib.load(LE_PATH)

def extract_features(sar_obs, monthly_ndvi, monthly_ndre):
    """
    Extract 63-feature vector from SAR + optical time series.
    
    Args:
        sar_obs: list of SAR observation dicts with date, vh, vv
        monthly_ndvi: dict {month_int: ndvi_float}
        monthly_ndre: dict {month_int: ndre_float}
    
    Returns:
        numpy array of 63 features
    """
    # Build monthly SAR arrays
    monthly_vh_raw = {}
    monthly_vv_raw = {}
    for o in sar_obs:
        if not o.get("available"): continue
        try:
            month = int(o["date"].split("-")[1])
            vh = o.get("vh") or o.get("vh_mean")
            vv = o.get("vv") or o.get("vv_mean")
            if vh:
                if month not in monthly_vh_raw: monthly_vh_raw[month] = []
                monthly_vh_raw[month].append(float(vh))
            if vv:
                if month not in monthly_vv_raw: monthly_vv_raw[month] = []
                monthly_vv_raw[month].append(float(vv))
        except: continue

    vh   = np.array([np.mean(monthly_vh_raw.get(m,[])) if m in monthly_vh_raw else np.nan for m in range(1,13)])
    vv   = np.array([np.mean(monthly_vv_raw.get(m,[])) if m in monthly_vv_raw else np.nan for m in range(1,13)])
    ndvi = np.array([float(monthly_ndvi.get(m, 0)) for m in range(1,13)], dtype=float)
    ndre = np.array([float(monthly_ndre.get(m, 0)) for m in range(1,13)], dtype=float)

    for arr in [vh, vv, ndvi, ndre]: arr[arr==0]=np.nan
    ratio = np.where(vh>0, vv/vh, np.nan)

    # VV range from all SAR observations
    all_vv = [float(o.get("vv") or o.get("vv_mean",0)) for o in sar_obs if o.get("available") and (o.get("vv") or o.get("vv_mean"))]
    vv_range = max(all_vv)-min(all_vv) if len(all_vv)>3 else 0

    def smean(arr,months):
        vals=[arr[mo-1] for mo in months if not np.isnan(arr[mo-1])]
        return np.mean(vals) if vals else np.nan
    def sval(arr,month): return arr[month-1]
    def safe_skew(arr):
        v=[x for x in arr if not np.isnan(x)]
        return float(stats.skew(v)) if len(v)>=4 else np.nan
    def safe_kurtosis(arr):
        v=[x for x in arr if not np.isnan(x)]
        return float(stats.kurtosis(v)) if len(v)>=4 else np.nan
    def safe_integral(arr):
        v=[x for x in arr if not np.isnan(x)]
        return float(trapezoid(v)) if len(v)>=3 else np.nan
    def safe_corr(a,b):
        pairs=[(x,y) for x,y in zip(a,b) if not np.isnan(x) and not np.isnan(y)]
        if len(pairs)<4: return np.nan
        xa,ya=zip(*pairs)
        c=np.corrcoef(xa,ya)
        return float(c[0,1]) if not np.isnan(c[0,1]) else np.nan

    ndvi_jan=sval(ndvi,1);ndvi_feb=sval(ndvi,2);ndvi_mar=sval(ndvi,3)
    ndvi_apr=sval(ndvi,4);ndvi_may=sval(ndvi,5);ndvi_jun=sval(ndvi,6)
    ndvi_nov=sval(ndvi,11);ndvi_dec=sval(ndvi,12)
    ndre_jan=sval(ndre,1);ndre_feb=sval(ndre,2);ndre_mar=sval(ndre,3)
    ndre_apr=sval(ndre,4);ndre_may=sval(ndre,5);ndre_jun=sval(ndre,6)
    ratio_may=sval(ratio,5);ratio_apr=sval(ratio,4);ratio_jun=sval(ratio,6)
    vh_may=sval(vh,5);vh_apr=sval(vh,4);vh_jan=sval(vh,1)

    winter_mask=smean(ndvi,[1,2,3])
    winter_stability=1-np.nanstd([ndvi_jan,ndvi_feb,ndvi_mar])
    may_jun_delta=(ndvi_may-ndvi_jun) if not np.isnan(ndvi_may) and not np.isnan(ndvi_jun) else np.nan
    apr_jun_delta=(ndvi_jun-ndvi_apr) if not np.isnan(ndvi_jun) and not np.isnan(ndvi_apr) else np.nan
    jan_apr_contrast=(ndvi_jan-ndvi_apr) if not np.isnan(ndvi_jan) and not np.isnan(ndvi_apr) else np.nan
    ndre_jan_mar=smean(ndre,[1,2,3])
    ndre_may_jun=smean(ndre,[5,6]);ndvi_may_jun=smean(ndvi,[5,6])
    osr_filter=ndre_may_jun/(ndvi_may_jun+1e-6) if not np.isnan(ndre_may_jun) and not np.isnan(ndvi_may_jun) else np.nan
    osr_filter_apr=(ndre_apr/(ndvi_apr+1e-6)) if not np.isnan(ndre_apr) and not np.isnan(ndvi_apr) else np.nan
    osr_vshape=(ndre_feb-ndre_apr) if not np.isnan(ndre_feb) and not np.isnan(ndre_apr) else np.nan
    vh_volatility=np.nanstd(vh[3:7]) if np.sum(~np.isnan(vh[3:7]))>=2 else np.nan
    vv_volatility=np.nanstd(vv[3:7]) if np.sum(~np.isnan(vv[3:7]))>=2 else np.nan
    vh_vol_ratio=vh_volatility/(vv_volatility+1e-6) if not np.isnan(vh_volatility) and not np.isnan(vv_volatility) else np.nan
    ndre_diffs=np.diff(ndre)
    ndre_inflection=np.nanargmax(ndre_diffs) if np.sum(~np.isnan(ndre_diffs))>=2 else np.nan
    ndre_max_vel=np.nanmax(ndre_diffs) if np.sum(~np.isnan(ndre_diffs))>=2 else np.nan
    ndre_min_vel=np.nanmin(ndre_diffs) if np.sum(~np.isnan(ndre_diffs))>=2 else np.nan
    ndvi_diffs=np.diff(ndvi)
    ndvi_inflection=np.nanargmax(ndvi_diffs) if np.sum(~np.isnan(ndvi_diffs))>=2 else np.nan
    win_vh=smean(vh,[12,1,2]);spr_vh=smean(vh,[3,4,5])
    ratio_peak=np.nanargmax(ratio) if np.any(~np.isnan(ratio)) else np.nan
    ratio_mar=sval(ratio,3);ratio_winter=smean(ratio,[12,1,2])
    tillage_mar=(ratio_mar-ratio_winter) if not np.isnan(ratio_mar) and not np.isnan(ratio_winter) else np.nan
    ndvi_x_ratio_may=(ndvi_may*ratio_may) if not np.isnan(ndvi_may) and not np.isnan(ratio_may) else np.nan
    ndvi_x_ratio_apr=(ndvi_apr*ratio_apr) if not np.isnan(ndvi_apr) and not np.isnan(ratio_apr) else np.nan
    ndvi_x_ratio_jun=(ndvi_jun*ratio_jun) if not np.isnan(ndvi_jun) and not np.isnan(ratio_jun) else np.nan
    ndre_x_vh_may=(ndre_may*vh_may) if not np.isnan(ndre_may) and not np.isnan(vh_may) else np.nan
    ndre_x_vh_apr=(ndre_apr*vh_apr) if not np.isnan(ndre_apr) and not np.isnan(vh_apr) else np.nan
    ndvi_x_vh_jan=(ndvi_jan*vh_jan) if not np.isnan(ndvi_jan) and not np.isnan(vh_jan) else np.nan
    ndvi_slope=(ndvi_jun-ndvi_apr) if not np.isnan(ndvi_jun) and not np.isnan(ndvi_apr) else np.nan
    ratio_slope=(ratio_jun-ratio_apr) if not np.isnan(ratio_jun) and not np.isnan(ratio_apr) else np.nan
    coupled_slope=(ndvi_slope*ratio_slope) if not np.isnan(ndvi_slope) and not np.isnan(ratio_slope) else np.nan
    ratio_at_peak=ratio[np.nanargmax(ndvi)] if not np.all(np.isnan(ndvi)) else np.nan
    peak_month=float(np.nanargmax(ndvi)+1) if not np.all(np.isnan(ndvi)) else np.nan
    n_ndvi = len([v for v in monthly_ndvi.values() if v])

    return np.array([
        winter_mask,winter_stability,may_jun_delta,apr_jun_delta,
        jan_apr_contrast,ndre_jan_mar,osr_filter,osr_filter_apr,osr_vshape,
        vh_volatility,vv_volatility,vh_vol_ratio,
        ndre_inflection,ndre_max_vel,ndre_min_vel,ndvi_inflection,
        ndvi_jan,ndvi_feb,ndvi_mar,ndvi_apr,ndvi_may,ndvi_jun,ndvi_nov,ndvi_dec,
        smean(ndvi,[12,1,2]),smean(ndvi,[3,4,5]),
        ndre_jan,ndre_feb,ndre_mar,ndre_apr,ndre_may,ndre_jun,
        win_vh,spr_vh,vv_range,tillage_mar,ratio_peak,sval(vh,1),n_ndvi,
        ndvi_x_ratio_may,ndvi_x_ratio_apr,ndvi_x_ratio_jun,
        ndre_x_vh_may,ndre_x_vh_apr,ndvi_x_vh_jan,
        coupled_slope,ratio_at_peak,ratio_may,
        safe_skew(ndvi),safe_kurtosis(ndvi),safe_integral(ndvi),
        safe_skew(ndre),safe_kurtosis(ndre),safe_integral(ndre),
        safe_skew(vh),safe_kurtosis(vh),safe_integral(vh),
        safe_corr(ndvi,vh),safe_corr(ndvi,ratio),
        safe_corr(ndre,vh),safe_corr(ndvi,ndre),
        (sval(ndvi,10)/sval(ndvi,1)) if not np.isnan(sval(ndvi,10)) and not np.isnan(sval(ndvi,1)) and sval(ndvi,1)>0 else np.nan,
        peak_month,
    ], dtype=float)


def classify_crop(sar_obs, monthly_ndvi, monthly_ndre,
                  confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Classify crop type from satellite time series.

    Returns dict:
        crop_type:   identified crop or "Uncertain"
        confidence:  0-100 operational confidence
        probabilities: all class probabilities
        confident:   True if above threshold
        ranking:     sorted list of (crop, probability)
        method:      "CropFusion v18"
    """
    _load()
    features = extract_features(sar_obs, monthly_ndvi, monthly_ndre)
    X = features.reshape(1, -1)
    proba = _pipe.predict_proba(X)[0]
    classes = _le.classes_
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    best_crop = str(ranked[0][0])
    best_prob = float(ranked[0][1])
    confident = best_prob >= confidence_threshold

    return {
        "crop_type": best_crop if confident else "Uncertain",
        "top_crop": best_crop,
        "confidence": round(best_prob * 100),
        "confident": confident,
        "threshold_used": confidence_threshold,
        "ranking": [{"crop": str(c), "pct": round(float(p)*100)} for c,p in ranked],
        "method": "CropFusion v18 — SAR+NDVI+NDRE (90% acc @ 50% threshold)"
    }
