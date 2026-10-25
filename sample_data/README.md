# TerraSentinel AI — Calibration & Demonstration Datasets

## Overview
This directory contains calibrated demonstration benchmarks designed for live technical evaluation, algorithm verification, and hackathon presentation of **TerraSentinel AI**.

## Included Benchmark Datasets

| Dataset Key | Anomaly Type | Scenario / Target | Image Resolution | Synthetic Notice |
| :--- | :--- | :--- | :--- | :--- |
| `deforestation` | Deforestation & Canopy Disturbance | Fishbone logging roads & slash clear-cutting | 10 m / px | Synthetic Benchmark Model |
| `flood` | Flood Inundation & Water Expansion | Seasonal river overflow into floodplain | 15 m / px | Synthetic Benchmark Model |
| `mining` | Excavation & Exposed Soil | Open-cast terraced mineral pit expansion | 5 m / px | Synthetic Benchmark Model |
| `urban_construction` | Infrastructure Construction | Rectilinear warehouse & asphalt logistics facility | 5 m / px | Synthetic Benchmark Model |
| `unchanged_control` | No Significant Change | Stable canopy baseline with sensor noise | 10 m / px | Synthetic Benchmark Model |

## Scientific Integrity Statement
As required by the TerraSentinel AI scientific principles:
- These calibration pairs are **Synthetic Test Datasets** generated using multi-layer spectral synthesis, Perlin texture approximations, and geometric primitives.
- They are **explicitly labeled as synthetic** and are never presented as unverified actual satellite photography.
- They serve to demonstrate deterministic pixel-difference calculation, connected component segmentation, heuristic classification, and report generation in an offline environment without reliance on external download bandwidth.

## Real-World Imagery Acquisition Workflow
To analyze real operational satellite data with TerraSentinel AI:
1. **Copernicus Data Space Ecosystem (Sentinel-2 MSI)**:
   - Access: [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
   - Bands: True Color RGB (B04, B03, B02) at 10m Ground Sample Distance (GSD).
2. **NASA Worldview / Earthdata (Landsat 8/9 OLI)**:
   - Access: [worldview.earthdata.nasa.gov](https://worldview.earthdata.nasa.gov)
   - Bands: Natural color RGB (Bands 4, 3, 2) at 30m GSD.
3. **PlanetScope (Commercial Sub-Meter)**:
   - Access: [planet.com/explorer](https://www.planet.com/explorer)
   - Bands: High-resolution 3m ortho-scene RGB products.
