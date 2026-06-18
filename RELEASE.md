# v1.2-production — Release Notes
*June 18, 2026*

## What's in this release

### Inference
- CatBoost 7-class crop classifier (63.5% CV accuracy)
- Two-tier confidence routing: Tier1 ≥60% → 90% precision
- NDVI quality gate: rejects parcels with <7 optical months
- Honest Tier3 rejection — no forced predictions on bad data

### Performance
- SAR observation cache: 1.4s warm response (was 90s+)
- Parcel-keyed cache by PAR_LAB + season
- Deployment-safe relative paths (no /workspaces hardcoding)

### API
- /parcel_intelligence: CatBoost result in every response
- /crop_classify: ML-only endpoint
- field_variability 0.0 bug fixed
- CatBoost reuses pre-fetched SAR — no duplicate STAC calls

### Quality
- 8/8 regression tests passing
- Threshold sensitivity study: 60% gate = 82.2% precision
- Sensor ablation: NDRE > NDVI >> SAR for Irish crops

## Known limitations
- CatBoost trained on Oct 2024–Sep 2025 season only
- n_ndvi=5 parcels (heavily clouded) → Tier3 rejection (~8%)
- Cache resets on Render free tier restart
- Oats classification weak (31%) — biologically similar to Barley

## Next milestone (July 1, 2026)
- CDSE PU reset → re-extract 1000+ parcels
- NISAR L-band V1 expected → L-band SAR for Ireland
- Retrain CatBoost on 2025-2026 season data
- Target: 8-class accuracy 65%+
