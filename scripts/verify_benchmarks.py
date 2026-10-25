"""Verification of all 5 benchmark demo datasets."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.image_loader import load_image
from modules.change_detection import detect_changes
from modules.anomaly_classification import classify_changes

benchmarks = ['deforestation', 'flood', 'mining', 'urban_construction', 'unchanged_control']
for b_name in benchmarks:
    b, _ = load_image(f'sample_data/{b_name}/before.png')
    a, _ = load_image(f'sample_data/{b_name}/after.png')
    res = detect_changes(b, a)
    heur = classify_changes(b, a, res['binary_mask'], res['regions'], res['metrics'])
    cat = heur["primary_category"]
    conf = heur["confidence"]
    pct = res["metrics"]["change_percentage"]
    print(f"[{b_name.upper():20}] => {cat} ({conf} Confidence, {pct:.2f}% changed)")
