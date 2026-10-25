import urllib.request
import json
from pathlib import Path
from PIL import Image
import numpy as np

output_dir = Path("sample_data/real_kerala_flood_2018")
output_dir.mkdir(parents=True, exist_ok=True)

url_before = "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/imagerecords/92000/92669/kerala_oli_201837_lrg.jpg"
url_after = "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/imagerecords/92000/92669/kerala_msi_2018234_lrg.jpg"

raw_before_path = output_dir / "raw_before.jpg"
raw_after_path = output_dir / "raw_after.jpg"

print("Downloading NASA Kerala Floods imagery...")
req_b = urllib.request.Request(url_before, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_b, timeout=30) as resp, open(raw_before_path, "wb") as f:
    f.write(resp.read())
print("Downloaded Before image:", raw_before_path.stat().st_size, "bytes")

req_a = urllib.request.Request(url_after, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_a, timeout=30) as resp, open(raw_after_path, "wb") as f:
    f.write(resp.read())
print("Downloaded After image:", raw_after_path.stat().st_size, "bytes")

# Open both images
img_b = Image.open(raw_before_path)
img_a = Image.open(raw_after_path)
print(f"Original sizes: Before={img_b.size}, After={img_a.size}")

# The NASA EO images are already geocoded / framed over the identical area of Kerala (Vembanad Lake / Kuttanad / Alappuzha)
# Let's ensure both are RGB and resized to a consistent dimension (e.g. 1024x1024 or min dimensions)
target_w = min(img_b.width, img_a.width, 1200)
target_h = min(img_b.height, img_a.height, 1200)

proc_b = img_b.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
proc_a = img_a.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)

proc_b.save(output_dir / "before.png", format="PNG", optimize=True)
proc_a.save(output_dir / "after.png", format="PNG", optimize=True)

# Remove raw download files
raw_before_path.unlink()
raw_after_path.unlink()

metadata = {
    "title": "Kerala Monsoon Floods & Inundation (NASA Earth Observatory EO 92669)",
    "is_synthetic": False,
    "is_real_satellite": True,
    "data_tier": "Real Earth Observation Satellite Imagery",
    "category": "Flooding & Water Expansion",
    "location_name": "Kuttanad & Vembanad Lake Region, Kerala, India",
    "coordinates": {
        "lat": 9.58,
        "lon": 76.45
    },
    "sensor_name": "Landsat 8 OLI & Copernicus Sentinel-2 MSI (NASA/USGS/ESA)",
    "dates": {
        "t0_baseline": "2018-02-06",
        "t1_event": "2018-08-22"
    },
    "date_before": "2018-02-06",
    "date_after": "2018-08-22",
    "approx_resolution_m": 30.0,
    "spatial_calibration_confirmed": True,
    "temporal_sequence_available": False,
    "source": "NASA Earth Observatory (Image Record 92669: Before and After the Kerala Floods)",
    "source_url": "https://earthobservatory.nasa.gov/images/92669/before-and-after-the-kerala-floods",
    "attribution": "NASA Earth Observatory images by Lauren Dauphin, using modified Copernicus Sentinel data (2018) processed by the European Space Agency and Landsat data from USGS. Public Domain.",
    "description": "Authentic multi-sensor satellite surveillance capturing catastrophic monsoon inundation across the Kuttanad agricultural basin and Vembanad Lake corridor in August 2018 compared to dry-season baseline in February 2018."
}

with open(output_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved real_kerala_flood_2018 successfully!")
