import json
import os
import sys
import numpy as np
import warnings
import joblib
import math
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/crop-trajectory')
from extractors.fusion_extractor import extract_fusion_features

def gm(dic, m):
    if not dic or not isinstance(dic, dict):
        return np.nan
    v = dic.get(str(m), dic.get(m, 0))
    return float(v) if v and float(v) != 0.0 else np.nan

def interp_channel(arr, fallback_defaults):
    x = np.arange(12)
    mask = ~np.isnan(arr)
    if mask.sum() >= 2:
        return np.interp(x, x[mask], arr[mask])
    elif mask.sum() == 1:
        return np.full(12, arr[mask])
    return np.array(fallback_defaults).copy()

def predict_live_lpis_parcel(polygon_geometry, area_ha=5.0, perimeter_m=400.0):
    try:
        model = joblib.load('/workspaces/crop-trajectory/models/production_catboost_7class.pkl')
        le = joblib.load('/workspaces/crop-trajectory/models/encoder_7class.pkl')
        opt_indices = joblib.load('/workspaces/crop-trajectory/models/optimal_indices.pkl')
    except Exception as e:
        print('❌ Model load error:', e)
        return 'Unknown', 0.0

    try:
        feat_dict = extract_fusion_features(polygon_geometry, '2024-10-01', '2025-09-30')
        if not feat_dict or not isinstance(feat_dict, dict):
            print('⚠️ Extractor failed to return a valid mapping dictionary.')
            return 'Unknown', 0.0
    except Exception as e:
        print('❌ Extractor runtime crash:', e)
        return 'Unknown', 0.0

    nd_raw = np.array([gm(feat_dict.get('monthly_ndvi', {}), m) for m in range(1, 13)])
    nr_raw = np.array([gm(feat_dict.get('monthly_ndre', {}), m) for m in range(1, 13)])
    vh_raw = np.array([gm(feat_dict.get('monthly_vh', {}), m) for m in range(1, 13)])
    vv_raw = np.array([gm(feat_dict.get('monthly_vv', {}), m) for m in range(1, 13)])

    ndi = interp_channel(nd_raw, [0.4]*12)
    nri = interp_channel(nr_raw, [0.3]*12)
    vhi = interp_channel(vh_raw, [-17.0]*12)
    vvi = interp_channel(vv_raw, [-11.0]*12)

    compactness = (4.0 * np.pi * area_ha * 10000.0) / (perimeter_m ** 2 + 1e-6)
    elongation = perimeter_m / (4.0 * np.sqrt(area_ha * 10000.0) + 1e-6)
    geom_features = [area_ha, perimeter_m, compactness, elongation]

    X_full = np.array(list(ndi) + list(nri) + list(vhi) + list(vvi) + geom_features).reshape(1, -1)
    
    print('🔍 --- RECONSTRUCTED MATRIX TELEMETRY ---')
    print('Generated Dense Feature Vector Length:', X_full.shape, '(12 NDVI + 12 NDRE + 12 VH + 12 VV + 4 Geo)')
    print('Feature Vector Matrix Value Range   : Min =', round(float(np.nanmin(X_full)), 2), '| Max =', round(float(np.nanmax(X_full)), 2))
    
    X_opt = X_full[:, opt_indices]
    print('Applying Optimal Mask Index Filter    : Reduced shape down to:', X_opt.shape)

    # 5. FIX: Safe 2D matrix array index slicing to output clean string scalar conversions
    raw_pred = model.predict(X_opt)
    if hasattr(raw_pred, 'ndim') and raw_pred.ndim > 1:
        pred_idx = int(raw_pred[0][0])
    else:
        pred_idx = int(raw_pred[0])
        
    probs = model.predict_proba(X_opt)
    confidence = float(probs[0, pred_idx])
    
    predicted_crop = le.inverse_transform([pred_idx])[0]
    
    if confidence < 0.60:
        return 'Unknown', confidence
    return str(predicted_crop), confidence
