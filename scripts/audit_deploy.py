"""Audit script for Streamlit Community Cloud deployment readiness."""
import os
import re
from pathlib import Path

def check_windows_paths():
    print("--- CHECKING FOR HARDCODED WINDOWS PATHS ---")
    win_path = re.compile(r'[A-Za-z]:\\[a-zA-Z0-9_\-\\]+')
    found = 0
    for root, dirs, files in os.walk("."):
        if any(skip in root for skip in [".venv", "venv", "__pycache__", ".git"]):
            continue
        for f in files:
            if f.endswith((".py", ".json", ".toml", ".txt", ".md", ".yaml", ".yml")):
                p = os.path.join(root, f)
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as file:
                        for line_no, line in enumerate(file, 1):
                            if win_path.search(line):
                                print(f"  {p}:{line_no} -> {line.strip()[:100]}")
                                found += 1
                except Exception as e:
                    print(f"  Error reading {p}: {e}")
    if found == 0:
        print("  None found! All paths are relative or platform-independent.")
    else:
        print(f"  Found {found} potential hardcoded Windows paths.")

def check_file_sizes():
    print("\n--- CHECKING FILE SIZES (ASSETS & BENCHMARKS) ---")
    total_size = 0
    large_files = []
    for root, dirs, files in os.walk("."):
        if any(skip in root for skip in [".venv", "venv", "__pycache__", ".git"]):
            continue
        for f in files:
            p = os.path.join(root, f)
            sz = os.path.getsize(p)
            total_size += sz
            if sz > 10 * 1024 * 1024:  # > 10 MB
                large_files.append((p, sz / (1024 * 1024)))
    print(f"  Total repository size: {total_size / (1024 * 1024):.2f} MB")
    if large_files:
        print("  Large files (>10MB) that might block Git / Cloud deploy:")
        for lf, sz in large_files:
            print(f"    {lf}: {sz:.2f} MB")
    else:
        print("  No oversized files (>10MB). Clean footprint for Git and Streamlit Cloud.")

def check_secrets():
    print("\n--- CHECKING FOR SECRETS / CREDENTIALS / API KEYS ---")
    secret_patterns = [
        re.compile(r'(?i)(api[_-]?key|secret|password|auth[_-]?token|bearer)\s*[:=]\s*["\']([a-zA-Z0-9_\-]{16,})["\']'),
        re.compile(r'ghp_[a-zA-Z0-9]{36}'),
        re.compile(r'sk-[a-zA-Z0-9]{32,}'),
        re.compile(r'AKIA[0-9A-Z]{16}')
    ]
    found = 0
    for root, dirs, files in os.walk("."):
        if any(skip in root for skip in [".venv", "venv", "__pycache__", ".git"]):
            continue
        for f in files:
            if f in [".env.example"]:
                continue
            if f.endswith((".py", ".json", ".toml", ".txt", ".env")):
                p = os.path.join(root, f)
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as file:
                        for line_no, line in enumerate(file, 1):
                            for pattern in secret_patterns:
                                m = pattern.search(line)
                                if m:
                                    print(f"  POTENTIAL SECRET in {p}:{line_no}")
                                    found += 1
                except Exception as e:
                    print(f"  Error reading {p}: {e}")
    if found == 0:
        print("  Zero exposed secrets detected in repository files.")
    else:
        print(f"  WARNING: Found {found} potential exposed secrets!")

if __name__ == "__main__":
    check_windows_paths()
    check_file_sizes()
    check_secrets()
