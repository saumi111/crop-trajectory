import json
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def execute_v17_expanded_training():
    print("🚀 Initializing v17 Expanded Class Dataset Out-of-Fold Evaluation (586 Parcels)...")
    
    with open('/workspaces/crop-trajectory/data/dataset_merged.json') as f:
        data = json.load(f)
        
    # We include Potatoes in the active target mapping array now that we have 39 samples
    CM = {'Grassland':'Grassland','Barley':'Barley','Wheat':'Wheat','Oats':'Oats',
          'Oilseed Rape':'Oilseed Rape','Maize':'Maize','Beans':'Beans','Potatoes':'Potatoes'}
          
    data = [d for d in data if d.get('label') in CM]
    labels = [CM[d['label']] for d in data]
    
    def gm(dic, m):
        if not dic or not isinstance(dic, dict): return np.nan
        v = dic.get(str(m), dic.get(m, 0))
        return float(v) if v and float(v) != 0.0 else np.nan
        
    def interp(arr, fb):
        x = np.arange(12); mask = ~np.isnan(arr)
        if mask.sum() >= 2: return np.interp(x, x[mask], arr[mask])
        elif mask.sum() == 1: return np.full(12, arr[mask])
        return np.array(fb).copy()
        
    all_ndvis, all_ndres, all_vhs, all_vvs = [{m: [] for m in range(1, 13)} for _ in range(4)]
    for d in data:
        for m in range(1, 13):
            n, nr = gm(d.get('monthly_ndvi',{}), m), gm(d.get('monthly_ndre',{}), m)
            vh, vv = gm(d.get('monthly_vh',{}), m), gm(d.get('monthly_vv',{}), m)
            if not np.isnan(n): all_ndvis[m].append(n)
            if not np.isnan(nr): all_ndres[m].append(nr)
            if not np.isnan(vh): all_vhs[m].append(vh)
            if not np.isnan(vv): all_vvs[m].append(vv)
            
    f_nd = [np.median(all_ndvis[m]) if all_ndvis[m] else 0.4 for m in range(1, 13)]
    f_nr = [np.median(all_ndres[m]) if all_ndres[m] else 0.3 for m in range(1, 13)]
    f_vh = [np.median(all_vhs[m]) if all_vhs[m] else -17.0 for m in range(1, 13)]
    f_vv = [np.median(all_vvs[m]) if all_vvs[m] else -11.0 for m in range(1, 13)]
    
    X_f = []
    for d in data:
        ndi = interp(np.array([gm(d.get('monthly_ndvi', {}), m) for m in range(1, 13)]), f_nd)
        nri = interp(np.array([gm(d.get('monthly_ndre', {}), m) for m in range(1, 13)]), f_nr)
        vhi = interp(np.array([gm(d.get('monthly_vh', {}), m) for m in range(1, 13)]), f_vh)
        vvi = interp(np.array([gm(d.get('monthly_vv', {}), m) for m in range(1, 13)]), f_vv)
        a = float(d.get('area_ha', 5.0))
        p = float(d.get('perimeter_m', 400.0))
        c = (4.0 * np.pi * a * 10000.0) / (p ** 2 + 1e-6)
        el = p / (4.0 * np.sqrt(a * 10000.0) + 1e-6)
        X_f.append(list(ndi) + list(nri) + list(vhi) + list(vvi) + [a, p, c, el])
        
    X_f = np.array(X_f, dtype=np.float64)
    le = LabelEncoder(); y = le.fit_transform(labels)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_true, all_pred, all_conf = [], [], []
    
    print("🔄 Training CatBoost across 5 stratified out-of-fold partitions...")
    for train_idx, val_idx in cv.split(X_f, y):
        Xt_full, Xv_full = X_f[train_idx], X_f[val_idx]
        yt, yv = y[train_idx], y[val_idx]
        
        # Pull optimal baseline feature mask
        m_find = CatBoostClassifier(iterations=200, verbose=0, random_seed=42)
        m_find.fit(Xt_full, yt)
        idx = np.argsort(m_find.get_feature_importance())[::-1][:30]
        
        Xt, Xv = Xt_full[:, idx], Xv_full[:, idx]
        clf = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.06, loss_function="MultiClass", verbose=0, random_seed=42)
        clf.fit(Xt, yt)
        
        fold_preds = clf.predict(Xv).flatten()
        probs = clf.predict_proba(Xv)
        
        for i in range(len(val_idx)):
            p_idx = int(fold_preds[i])
            all_true.append(int(yv[i]))
            all_pred.append(p_idx)
            all_conf.append(float(probs[i, p_idx]))
            
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_conf = np.array(all_conf)
    
    print("\n📊 v17 Expanded Dataset Ingestion Metrics Matrix:")
    print("-" * 70)
    print(classification_report(all_true, all_pred, target_names=list(le.classes_)))
    print("-" * 70)
    
    # Track Tier 1 precision delivery metrics under high-volume conditions
    print("\n📈 Automated Delivery Tier Gating Tradeoffs Profile:")
    for t in [0.60, 0.55]:
        mask = all_conf >= t
        count = np.sum(mask)
        prec = np.mean(all_true[mask] == all_pred[mask]) * 100.0 if count > 0 else 0.0
        print(f"  Gate {int(t*100)}% -> Accepted Volume: {count:<3} parcels ({(count/len(all_true))*100:.1f}%) | Tier-1 Precision: {prec:.1f}%")

if __name__ == "__main__":
    execute_v17_expanded_training()
