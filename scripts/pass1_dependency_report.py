#!/usr/bin/env python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis.dependency_probe import probe_pass1_dependencies


def main() -> None:
    probe = probe_pass1_dependencies()
    print("Pass1 dependency availability:")
    for dep, ok in sorted(probe.availability.items()):
        print(f"- {dep}: {'OK' if ok else 'MISSING'}")

    missing = probe.missing_required()
    if missing:
        print("\nRequired for full ported Pass1 runtime missing:", ", ".join(missing))
    else:
        print("\nAll required dependencies for full ported Pass1 runtime are available.")


if __name__ == "__main__":
    main()
