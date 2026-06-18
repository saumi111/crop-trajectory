import json
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/workspaces/crop-trajectory')
from tools.inference_driver import predict_from_observations

def calculate_polygon_perimeter(coords):
    """Calculates basic planar boundary perimeter lengths from coordinate outer rings"""
    try:
        pts = coords if len(coords) == 1 and isinstance(coords, list) else coords
        total_p = 0.0
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            total_p += np.sqrt((float(p2) - float(p1))**2 + (float(p2) - float(p1))**2) * 111320.0
        return max(total_p, 100.0)
    except:
        return 400.0

def execute_batch_geojson_classification(input_path, output_path):
    print("🎬 Running Production Two-Tier GeoJSON Batch Processor with Analyst Queue Isolation...")
    if not os.path.exists(input_path):
        print(f"❌ Input file missing at: {input_path}")
        return
        
    with open(input_path, "r") as f:
        geojson_data = json.load(f)
        
    features = geojson_data.get("features", [])
    total_features = len(features)
    print(f"📌 Layer Parsing Successful. Found {total_features} features to process.")
    
    auto_accept_features = []
    review_queue_features = []
    t1_count, t2_count, t3_count = 0, 0, 0
    
    for idx, feature in enumerate(features):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        fid = props.get("parcel_id", props.get("PARCEL_ID", f"BTI-{idx:03}"))
        
        if geom.get("type") not in ["Polygon", "MultiPolygon"] or "coordinates" not in geom:
            continue
            
        poly_coords = geom["coordinates"]
        area_ha = float(props.get("area_ha", props.get("AREA_HA", 4.5)))
        perimeter_m = calculate_polygon_perimeter(poly_coords)
        
        # Execute the optimized, observation-aware inference driver
        res = predict_from_observations(
            poly_coords, 
            client_id=os.environ.get("COP0_ID", "DUMMY_ID"),
            client_secret=os.environ.get("COP0_SECRET", "DUMMY_SECRET"),
            sar_observations=None,
            area_ha=area_ha, 
            perimeter_m=perimeter_m
        )
        
        pred_crop_clean = res["predicted_crop"]
        confidence = res["confidence_pct"] / 100.0
        tier = res["tier"]
        
        new_props = dict(props)
        new_props["raw_prediction"] = pred_crop_clean
        new_props["inference_confidence"] = float(res["confidence_pct"])
        new_props["delivery_tier"] = tier
        new_props["automated_delivery"] = res["automated_delivery"]
        
        # MULTI-TIER OPERATIONAL INGESTION ROUTING
        if tier == "Tier1":
            new_props["crop_prediction"] = pred_crop_clean
            t1_count += 1
            status_str = f"🚀 TIER 1 - AUTO-ACCEPT ({pred_crop_clean} @ {res['confidence_pct']}%)"
            
            new_feature = {"type": "Feature", "geometry": geom, "properties": new_props}
            auto_accept_features.append(new_feature)
            
        elif tier == "Tier2":
            # Expose the raw prediction context safely inside the review workspace properties
            new_props["crop_prediction"] = f"{pred_crop_clean} (Low Confidence Hint)"
            t2_count += 1
            status_str = f"⚠️ TIER 2 - ROUTED TO ANALYST REVIEW QUEUE ({pred_crop_clean} @ {res['confidence_pct']}%)"
            
            new_feature = {"type": "Feature", "geometry": geom, "properties": new_props}
            review_queue_features.append(new_feature)
            
        else:
            new_props["crop_prediction"] = "Unknown"
            t3_count += 1
            status_str = f"🛑 TIER 3 - REJECTED ({res['confidence_pct']}% certainty)"
            
            new_feature = {"type": "Feature", "geometry": geom, "properties": new_props}
            auto_accept_features.append(new_feature)
            
        print(f"  [{idx+1}/{total_features}] Parcel ID: {fid:<12} | {status_str}")
        
    # Write the main high-confidence/rejected delivery dataset
    with open(output_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": auto_accept_features}, f, indent=2)
        
    # Export borderline fields to the Analyst Review Queue file
    review_queue_path = "/workspaces/crop-trajectory/data/review_queue.geojson"
    with open(review_queue_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": review_queue_features}, f, indent=2)
        
    print("\n📊 Two-Tier Batch Process Operational Summary:")
    print(f"  Total Extracted Input Geometries           : {total_features}")
    print(f"  Tier 1 Production Automated Deliveries     : {t1_count} parcels")
    print(f"  Tier 2 Isolated for Manual Analyst Review : {t2_count} parcels (Saved -> {review_queue_path})")
    print(f"  Tier 3 Complete Uncertainty Gated Blocks   : {t3_count} parcels")
    print(f"💾 Primary annotated vector map layer saved to: {output_path}")

if __name__ == "__main__":
    execute_batch_geojson_classification("/workspaces/crop-trajectory/data/real_parcel.json", "/workspaces/crop-trajectory/data/classified_parcels.geojson")
