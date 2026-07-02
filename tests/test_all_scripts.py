from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_DIR = PROJECT_ROOT / "scripts"

py_files = sorted(SCRIPT_DIR.glob("*.py"))

for script in py_files:
    if script.name == "0.download_datasets.py":
        print(f"Skipping: {script.name} (to avoid downloading datasets during tests)")
        continue
    
    print(f"Running: {script.name}")
    
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True
    )

    print("STDOUT:")
    print(result.stdout)

    print("STDERR:")
    print(result.stderr)

    print("=" * 50)