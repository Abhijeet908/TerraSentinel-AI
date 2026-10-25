"""Inspect indicators for mining and urban construction."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.image_loader import load_image
from modules.change_detection import detect_changes
from modules.anomaly_classification import classify_changes

for name in ['mining', 'urban_construction']:
    b, _ = load_image(f'sample_data/{name}/before.png')
    a, _ = load_image(f'sample_data/{name}/after.png')
    res = detect_changes(b, a)
    heur = classify_changes(b, a, res['binary_mask'], res['regions'], res['metrics'])
    print(f"--- {name} ---")
    print("Primary:", heur["primary_category"])
    print("Possible:", heur["possible_categories"])
    print("Indicators:", heur["heuristic_indicators"])
