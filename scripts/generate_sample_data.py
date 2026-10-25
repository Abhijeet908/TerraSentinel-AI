"""Generates explicitly labeled synthetic calibration benchmarks for TerraSentinel AI."""
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "sample_data"
SAMPLE_DIR.mkdir(exist_ok=True)

def generate_natural_noise(shape, scale=10):
    h, w = shape[:2]
    small_h, small_w = max(4, h // scale), max(4, w // scale)
    rand = np.random.uniform(0, 1, (small_h, small_w)).astype(np.float32)
    pil_rand = Image.fromarray((rand * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
    return np.array(pil_rand).astype(np.float32) / 255.0

def create_synthetic_flood_temporal():
    folder = SAMPLE_DIR / "synthetic_flood_temporal"
    folder.mkdir(exist_ok=True)
    w, h = 640, 640

    # Base valley
    base = np.zeros((h, w, 3), dtype=np.float32)
    grid_size = 80
    for r in range(0, h, grid_size):
        for c in range(0, w, grid_size):
            h_val = np.random.uniform(0.7, 1.3)
            base[r:r+grid_size, c:c+grid_size, 0] = 70 * h_val
            base[r:r+grid_size, c:c+grid_size, 1] = 130 * h_val
            base[r:r+grid_size, c:c+grid_size, 2] = 50 * h_val

    # River channel
    pil_river = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
    draw = ImageDraw.Draw(pil_river)
    draw.line([(0, 320), (200, 340), (400, 300), (640, 330)], fill=255, width=16)
    river_arr = np.array(pil_river) > 0
    base[river_arr, 0] = 20
    base[river_arr, 1] = 50
    base[river_arr, 2] = 80

    t0_before = np.clip(base + np.random.normal(0, 3, base.shape), 0, 255).astype(np.uint8)

    # T1: Inundated Flood Peak
    flood_base = base.copy()
    flood_mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
    draw_f = ImageDraw.Draw(flood_mask)
    draw_f.ellipse([80, 160, 560, 480], fill=255)
    flood_arr = np.array(flood_mask.filter(ImageFilter.GaussianBlur(10))) > 80

    flood_base[flood_arr, 0] = 28
    flood_base[flood_arr, 1] = 48
    flood_base[flood_arr, 2] = 68
    t1_flood = np.clip(flood_base + np.random.normal(0, 4, flood_base.shape), 0, 255).astype(np.uint8)

    # T2: Receded Flood Waters (Transient reversal - water returns to near-channel with wet mud)
    t2_receded = base.copy()
    mud_mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
    draw_m = ImageDraw.Draw(mud_mask)
    draw_m.ellipse([160, 240, 480, 400], fill=255)
    mud_arr = np.array(mud_mask.filter(ImageFilter.GaussianBlur(8))) > 100
    # Mud has slightly darker soil hue, but most vegetation recovered
    t2_receded[mud_arr, 0] = 65
    t2_receded[mud_arr, 1] = 85
    t2_receded[mud_arr, 2] = 45
    t2_after = np.clip(t2_receded + np.random.normal(0, 3, t2_receded.shape), 0, 255).astype(np.uint8)

    Image.fromarray(t0_before).save(folder / "before.png")
    Image.fromarray(t1_flood).save(folder / "after.png")
    Image.fromarray(t0_before).save(folder / "t0_baseline.png")
    Image.fromarray(t1_flood).save(folder / "t1_event.png")
    Image.fromarray(t2_after).save(folder / "t2_recovery.png")

    meta = {
        "title": "Brahmaputra Basin Seasonal Inundation & Recession",
        "is_synthetic": True,
        "is_real_satellite": False,
        "data_tier": "Synthetic Calibration Benchmark",
        "category": "Flooding & Water-Body Changes",
        "location_name": "Assam Floodplain Basin Simulation",
        "sensor_name": "Synthetic 3-Band Optical Calibration Model",
        "date_before": "2024-04-10",
        "date_after": "2024-07-22",
        "dates": {
            "t0_baseline": "2024-04-10 (Pre-Monsoon)",
            "t1_event": "2024-07-22 (Peak Inundation)",
            "t2_recovery": "2024-10-15 (Post-Monsoon Recession)"
        },
        "approx_resolution_m": 15.0,
        "spatial_calibration_confirmed": False,
        "temporal_sequence_available": True,
        "description": "Synthetic benchmark modeling 3-stage temporal dynamics: dry floodplain baseline -> catastrophic monsoon inundation -> seasonal water recession (transient reversal)."
    }
    with open(folder / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def create_synthetic_mining():
    folder = SAMPLE_DIR / "synthetic_mining"
    folder.mkdir(exist_ok=True)
    w, h = 640, 640

    noise = generate_natural_noise((h, w), scale=14)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, 0] = 55 + 30 * noise
    base[:, :, 1] = 100 + 40 * noise
    base[:, :, 2] = 45 + 20 * noise

    before_img = np.clip(base + np.random.normal(0, 3, base.shape), 0, 255).astype(np.uint8)

    after_base = base.copy()
    pit_mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
    draw_p = ImageDraw.Draw(pit_mask)
    draw_p.polygon([(180, 180), (450, 160), (510, 400), (320, 480), (160, 360)], fill=255)
    pit_arr = np.array(pit_mask.filter(ImageFilter.GaussianBlur(4))) > 100

    after_base[pit_arr, 0] = 190
    after_base[pit_arr, 1] = 125
    after_base[pit_arr, 2] = 70

    draw_p.polygon([(240, 230), (390, 220), (430, 360), (310, 410), (220, 320)], fill=150)
    inner_arr = (np.array(pit_mask) == 150)
    after_base[inner_arr, 0] = 160
    after_base[inner_arr, 1] = 95
    after_base[inner_arr, 2] = 50

    after_img = np.clip(after_base + np.random.normal(0, 4, after_base.shape), 0, 255).astype(np.uint8)

    Image.fromarray(before_img).save(folder / "before.png")
    Image.fromarray(after_img).save(folder / "after.png")

    meta = {
        "title": "Open-Cast Pit Excavation & Exposed Soil",
        "is_synthetic": True,
        "is_real_satellite": False,
        "data_tier": "Synthetic Calibration Benchmark",
        "category": "Excavation & Exposed Soil",
        "location_name": "Pilbara Mineral Province Simulation",
        "sensor_name": "Synthetic 3-Band Optical Calibration Model",
        "date_before": "2023-11-05",
        "date_after": "2024-05-18",
        "approx_resolution_m": 5.0,
        "spatial_calibration_confirmed": False,
        "temporal_sequence_available": False,
        "description": "Synthetic benchmark modeling open-cast terraced excavation exposing subsoil minerals."
    }
    with open(folder / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def create_synthetic_control():
    folder = SAMPLE_DIR / "synthetic_unchanged_control"
    folder.mkdir(exist_ok=True)
    w, h = 640, 640

    noise = generate_natural_noise((h, w), scale=10)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, 0] = 60 + 25 * noise
    base[:, :, 1] = 110 + 35 * noise
    base[:, :, 2] = 55 + 20 * noise

    before_img = np.clip(base + np.random.normal(0, 2.0, base.shape), 0, 255).astype(np.uint8)
    after_img = np.clip(base + np.random.normal(0, 2.0, base.shape) + 1.0, 0, 255).astype(np.uint8)

    Image.fromarray(before_img).save(folder / "before.png")
    Image.fromarray(after_img).save(folder / "after.png")

    meta = {
        "title": "Zero-Anomaly Baseline Stability Control",
        "is_synthetic": True,
        "is_real_satellite": False,
        "data_tier": "Synthetic Calibration Benchmark",
        "category": "No Significant Change (Control)",
        "location_name": "Serengeti Conservation Buffer Simulation",
        "sensor_name": "Synthetic 3-Band Optical Calibration Model",
        "date_before": "2024-03-01",
        "date_after": "2024-03-15",
        "approx_resolution_m": 10.0,
        "spatial_calibration_confirmed": False,
        "temporal_sequence_available": False,
        "description": "Synthetic control benchmark with zero genuine land-cover modification and standard sensor noise to verify false-positive rejection."
    }
    with open(folder / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    create_synthetic_flood_temporal()
    create_synthetic_mining()
    create_synthetic_control()
    print("Synthetic benchmarks generated and documented.")
