"""
Run all data generators.
    python simulations/generate_all.py
"""

import subprocess
import sys
from pathlib import Path

GENERATORS = [
    "heat/generate_data.py",
    "wave/generate_data.py",
    "laplace/generate_data.py",
    "parabolic/generate_data.py",
]

base = Path(__file__).parent

for script in GENERATORS:
    path = base / script
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print('='*60)
    result = subprocess.run([sys.executable, str(path)], check=False)
    if result.returncode != 0:
        print(f"ERROR: {script} failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

print("\n✓ All datasets generated.")
