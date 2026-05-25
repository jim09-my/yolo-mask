"""Download the Face Mask Detection dataset from Kaggle and extract it to data/raw/."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def install_kagglehub():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])


def main():
    project_root = Path(__file__).resolve().parent.parent
    dest = project_root / "data" / "raw"

    try:
        import kagglehub  # noqa: F401
    except ImportError:
        print("Installing kagglehub...")
        install_kagglehub()
        import kagglehub

    print(f"Downloading andrewmvd/face-mask-detection to {dest} ...")
    archive_path = kagglehub.dataset_download(
        "andrewmvd/face-mask-detection", path=dest, force_download=True
    )
    archive_path = Path(archive_path)

    # The dataset contains a zip file inside the downloaded directory — extract it
    zip_files = list(archive_path.rglob("*.zip"))
    if zip_files:
        print(f"Extracting {len(zip_files)} archive(s)...")
        for zf in zip_files:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dest)
            zf.unlink()  # remove the zip after extraction
        print("Extraction complete.")

    # Clean up any leftover empty subdirectories from kagglehub
    for subdir in sorted(dest.rglob("*"), reverse=True):
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()

    print("Done. Files:")
    for item in sorted(dest.iterdir()):
        print(f"  {item.name}")


if __name__ == "__main__":
    main()
