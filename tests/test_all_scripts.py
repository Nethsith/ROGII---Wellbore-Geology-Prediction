from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"

py_files = sorted(SCRIPT_DIR.glob("*.py"))

failed_scripts = []

for script in py_files:
    print(f"\nRunning: {script.name}")

    result = subprocess.run(
        [sys.executable, str(script)],
        text=True
    )

    if result.returncode != 0:
        print(f"❌ FAILED: {script.name}")
        failed_scripts.append(script.name)
    else:
        print(f"✅ PASSED: {script.name}")

# 🚨 Fail CI if any script failed
if failed_scripts:
    print("\nFailed scripts:")
    for s in failed_scripts:
        print("-", s)
    sys.exit(1)

print("\n🎉 All scripts passed!")