import urllib.request
import re
import json
from pathlib import Path
from PIL import Image

output_dir = Path("sample_data/real_california_wildfire")
output_dir.mkdir(parents=True, exist_ok=True)

# NASA Earth Observatory Camp Fire / Paradise California Record 144225
url = "https://earthobservatory.nasa.gov/images/144225/historic-wildfire-in-california"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
        imgs = re.findall(r'https?://[^\s"\'<>]+\.(?:jpg|png)', html)
        print("Total images found on page:", len(imgs))
        camp_imgs = [i for i in imgs if any(k in i.lower() for k in ["camp", "paradise", "144225", "fire"])]
        print("Matching wildfire URLs:", len(camp_imgs))
        for ci in set(camp_imgs):
            print(" -", ci)
except Exception as e:
    print("Fetch error:", e)
