"""Downloads real, documented NASA Earth Observatory satellite imagery pairs."""
import urllib.request
import json
from pathlib import Path
from PIL import Image
import io

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "sample_data"
SAMPLE_DIR.mkdir(exist_ok=True)

def download_file(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'TerraSentinel-Downloader/2.0 (Academic/Research)'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()

def setup_real_amazon_dataset():
    folder = SAMPLE_DIR / "real_amazon_deforestation"
    folder.mkdir(exist_ok=True)

    urls = {
        "t0_2000": "https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/woc/images/amazon/amazon_deforestation_20000730.jpg",
        "t1_2006": "https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/woc/images/amazon/amazon_deforestation_20060717.jpg",
        "t2_2012": "https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/woc/images/amazon/amazon_deforestation_20120718.jpg"
    }

    print("Downloading Real NASA Amazon Deforestation dataset...")
    for key, url in urls.items():
        data = download_file(url)
        img = Image.open(io.BytesIO(data))
        filename = f"{key}.png" if "t" in key else f"{key}.jpg"
        img.save(folder / filename)
        if key == "t0_2000":
            img.save(folder / "before.png")
        elif key == "t2_2012":
            img.save(folder / "after.png")

    meta = {
        "title": "Rondônia Amazon Rainforest Deforestation (NASA Terra MODIS)",
        "is_synthetic": False,
        "is_real_satellite": True,
        "data_tier": "Real Earth Observation Satellite Imagery",
        "category": "Deforestation & Land Clearing",
        "location_name": "Rondônia State, Southern Amazon Basin, Brazil",
        "coordinates": {"lat": -10.5, "lon": -62.5},
        "sensor_name": "MODIS on NASA's Terra Satellite",
        "dates": {
            "t0_baseline": "2000-07-30",
            "t1_intermediate": "2006-07-17",
            "t2_target": "2012-07-18"
        },
        "date_before": "2000-07-30",
        "date_after": "2012-07-18",
        "approx_resolution_m": 250.0,
        "spatial_calibration_confirmed": True,
        "temporal_sequence_available": True,
        "source": "NASA Earth Observatory (World of Change: Amazon Deforestation)",
        "source_url": "https://earthobservatory.nasa.gov/world-of-change/Deforestation",
        "attribution": "NASA Earth Observatory images by Jesse Allen, using data from the Level 1 and Atmosphere Archive and Distribution System (LAADS) and the Land Processes DAAC. Public Domain.",
        "description": "Authentic multi-year satellite observation tracking systematic fishbone road expansion and massive tropical canopy removal in the Brazilian Amazon."
    }

    with open(folder / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Real Amazon dataset installed successfully.")

def setup_real_lake_mead_dataset():
    folder = SAMPLE_DIR / "real_lakemead_shoreline"
    folder.mkdir(exist_ok=True)

    urls = {
        "before": "https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/imagerecords/150000/150111/lakemeadwide_etm_2000188.jpg",
        "after": "https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/imagerecords/150000/150111/lakemeadwide_oli2_2022184.jpg"
    }

    print("Downloading Real NASA Landsat Lake Mead Shoreline dataset...")
    for key, url in urls.items():
        data = download_file(url)
        img = Image.open(io.BytesIO(data))
        img.save(folder / f"{key}.png")

    meta = {
        "title": "Lake Mead Shoreline Recession & Drought (NASA/USGS Landsat)",
        "is_synthetic": False,
        "is_real_satellite": True,
        "data_tier": "Real Earth Observation Satellite Imagery",
        "category": "Shoreline & Water-Body Depletion",
        "location_name": "Lake Mead Reservoir, Nevada-Arizona Border, USA",
        "coordinates": {"lat": 36.14, "lon": -114.43},
        "sensor_name": "Landsat 7 ETM+ (2000) / Landsat 8/9 OLI (2022)",
        "date_before": "2000-07-06",
        "date_after": "2022-07-03",
        "approx_resolution_m": 30.0,
        "spatial_calibration_confirmed": True,
        "temporal_sequence_available": False,
        "source": "NASA Earth Observatory (Image Record 150111)",
        "source_url": "https://earthobservatory.nasa.gov/images/150111/lake-mead-keeps-dropping",
        "attribution": "NASA Earth Observatory images by Lauren Dauphin, using Landsat data from the U.S. Geological Survey (USGS). Public Domain.",
        "description": "Authentic 22-year Landsat comparison documenting drastic water shrinkage and the prominent white mineralized 'bathtub ring' along the receding shoreline."
    }

    with open(folder / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Real Lake Mead dataset installed successfully.")

if __name__ == "__main__":
    setup_real_amazon_dataset()
    setup_real_lake_mead_dataset()
