import importlib
import sys

from offline.config import DATA_FEATURES, DATA_PROCESSED, DATA_RAW, MODELS_DIR, SAMPLE_RATE

PACKAGES = [
    "numpy", "pandas", "scipy", "librosa", "soundfile", "sklearn",
    "matplotlib", "seaborn", "tqdm", "yaml", "torch", "torchaudio",
]


def check_packages() -> bool:
    """Try to import every required package and print its version."""
    all_ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"[ OK ] {name:<12} {version}")
        except ImportError:
            print(f"[FAIL] {name:<12} not installed")
            all_ok = False
    return all_ok


def check_folders() -> bool:
    """Confirm that the data and model folders exist."""
    all_ok = True
    for folder in (DATA_RAW, DATA_PROCESSED, DATA_FEATURES, MODELS_DIR):
        if folder.exists():
            print(f"[ OK ] folder {folder}")
        else:
            print(f"[FAIL] folder missing: {folder}")
            all_ok = False
    return all_ok


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    print(f"Sample rate from config: {SAMPLE_RATE} Hz\n")
    packages_ok = check_packages()
    print()
    folders_ok = check_folders()
    print()
    if packages_ok and folders_ok:
        print("Environment is ready.")
    else:
        print("Fix the FAIL lines above, then run this again.")


if __name__ == "__main__":
    main()