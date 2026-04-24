#!/usr/bin/env python3
"""scan_folders.py — Scan Mendeley and Zenodo folders. Print every file."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BASE_DIR, MENDELEY_DIR, ZENODO_DIR, PROJECT_DIR

for path in [MENDELEY_DIR, ZENODO_DIR]:
    assert os.path.exists(path), f"ABORT: Path not found: {path}"
os.makedirs(PROJECT_DIR, exist_ok=True)
print(f"Mendeley: {MENDELEY_DIR}")
print(f"Zenodo:   {ZENODO_DIR}")
print(f"Project:  {PROJECT_DIR}")

for root_folder in [MENDELEY_DIR, ZENODO_DIR]:
    print(f"\n{'='*60}")
    print(f"FOLDER: {root_folder}")
    print(f"{'='*60}")
    for root, dirs, files in os.walk(root_folder):
        level = root.replace(root_folder, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            print(f"{indent}  {f}  ({size/1e6:.1f} MB)")
