from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"

py_files = sorted(SCRIPT_DIR.glob("*.py"))

for script in py_files:
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