import json, sys, os, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspaces/crop-trajectory")
from tools.inference_driver import predict_live_lpis_parcel

def calculate_polygon_perimeter(coords):
    """Calculates basic planar boundary perimeter lengths from coordinate outer rings"""
    try:
        pts = coords[0] if len(coords) == 1 and isinstance(coords[0], list) else coords
        total_p = 0.0
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            # Convert approximate degree distance differences to planar meters
            total_p += np.sqrt((float(p2[0]) - float(p1[0]))**2 + (float(p2[1]) - float(p1[1]))**2) * 111320.0
        return max(total_p, 100.0)
    except: return 400.0

def execute_batch_geojson_classification(input_path, output_path):
    print(f"🎬 Initializing Automated GeoJSON Batch Processing Pipeline...")
    print(f"   Input Layer Asset : {input_path}")
    print(f"   Output Target Path: {output_path}\n")
    
    if not os.path.exists(input_path):
        print(f"❌ Execution Error: Input file missing at {input_path}")
        return
        
    try:
        with open(input_path, "r") as f:
            geojson_data = json.load(f)
    except Exception as e:
        print(f"❌ Execution Error: Failed to parse input GeoJSON. Details: {e}")
        return
        
    features = geojson_data.get("features", [])
    total_features = len(features)
    print(f"📌 Layer Parsing Successful. Found {total_features} total features to classify.")
    print("  " + "-"*70)
    
    processed_features = []
    success_count, gated_count = 0, 0
    
    for idx, feature in enumerate(features):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        fid = props.get("parcel_id", props.get("PARCEL_ID", f"BTI-{idx:03}"))
        
        if geom.get("type") not in ["Polygon", "MultiPolygon"] or "coordinates" not in geom:
            print(f"  [SKIP] Feature {fid:<12} | Status: Skipped due to unsupported non-polygon vector topology.")
            continue
            
        poly_coords = geom["coordinates"]
        
        # Extract true area metadata if present, fallback safely using default constants
        area_ha = float(props.get("area_ha", props.get("AREA_HA", 4.5)))
        perimeter_m = calculate_polygon_perimeter(poly_coords)
        
        print(f"🛰️  Processing feature [{idx+1}/{total_features}] | Parcel ID: {fid}...")
        pred_crop, confidence = predict_live_lpis_parcel(poly_coords, area_ha=area_ha, perimeter_m=perimeter_m)
        
        # Inject the model classifications back into the feature metadata properties mapping
        new_props = dict(props)
        new_props["predicted_crop"] = str(pred_crop)
        new_props["inference_confidence"] = float(round(confidence * 100, 2))
        
        if str(pred_crop) == "Unknown":
            new_props["delivery_status"] = "Gated (High Uncertainty)"
            gated_count += 1
            status_str = f"⚠️ GATED ({confidence*100:.1f}% confidence)"
        else:
            new_props["delivery_status"] = "Accepted"
            success_count += 1
            status_str = f"✅ ACCEPTED ({pred_crop} @ {confidence*100:.1f}% certainty)"
            
        print(f"     Result -> {status_str}")
        
        new_feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": new_props
        }
        processed_features.append(new_feature)
        
    # 4. Compile output collection and save to disk
    output_geojson = {
        "type": "FeatureCollection",
        "features": processed_features
    }
    
    with open(output_path, "w") as f:
        json.dump(output_geojson, f, indent=2)
        
    print("\n📊 Batch Process Complete Performance Summary:")
    print(f"  Total Extracted Input Geometries: {total_features}")
    print(f"  Cleared Confidence Gate Filters: {success_count} parcels")
    print(f"  Flagged as Unknown (Gated)      : {gated_count} parcels")
    print(f"💾 Annotated map layer written successfully to: {output_path}")

if __name__ == "__main__":
    # Simple baseline runtime verification parameters
    mock_input = "/workspaces/crop-trajectory/data/real_parcel.json"
    mock_output = "/workspaces/crop-trajectory/data/classified_parcels.geojson"
    if os.path.exists(mock_input):
        execute_batch_geojson_classification(mock_input, mock_output)
    else:
        print("🚀 Batch processing utility module initialized smoothly. Interface ready for operation.")