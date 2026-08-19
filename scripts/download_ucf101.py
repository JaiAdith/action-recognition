"""
Cross-platform UCF101 dataset downloader.

Downloads and extracts:
- UCF101 video dataset (~6.5 GB, 13,320 videos across 101 action classes)
- Official train/test split annotations

Works on Windows, Linux, and macOS.

Usage:
    python scripts/download_ucf101.py [--data_dir ./data]
"""

import argparse
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


UCF101_URLS = [
    "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
    "https://storage.googleapis.com/thumos14_files/UCF101_videos.rar",
]
SPLITS_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"


class DownloadProgressBar:
    def __init__(self, filename):
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size > 0:
            percent = int(block_num * block_size * 100 / total_size)
            percent = min(percent, 100)
            if percent != self.last_percent:
                mb_done = block_num * block_size / 1e6
                mb_total = total_size / 1e6
                print(f"\r  Downloading {self.filename}: {percent}% ({mb_done:.0f}/{mb_total:.0f} MB)", end="", flush=True)
                self.last_percent = percent
        else:
            mb_done = block_num * block_size / 1e6
            print(f"\r  Downloading {self.filename}: {mb_done:.0f} MB", end="", flush=True)


def download_file(url, dest_path, max_retries=5):
    """Download a file with progress bar and resume support."""
    filename = Path(dest_path).name
    print(f"\n  URL: {url}")
    ctx = ssl._create_unverified_context()

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            req = urllib.request.Request(url)
            if downloaded > 0:
                req.add_header("Range", f"bytes={downloaded}-")
                print(f"  Resuming from {downloaded / 1e6:.0f} MB (attempt {attempt})...")

            response = urllib.request.urlopen(req, context=ctx, timeout=60)
            total_size = int(response.headers.get("Content-Length", 0)) + downloaded
            block_size = 1024 * 1024  # 1 MB

            with open(dest_path, "ab" if downloaded > 0 else "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = int(downloaded * 100 / total_size) if total_size else 0
                    print(f"\r  Downloading {filename}: {pct}% ({downloaded / 1e6:.0f}/{total_size / 1e6:.0f} MB)",
                          end="", flush=True)

            print()

            # Verify download is complete
            if total_size > 0 and downloaded < total_size:
                raise IOError(
                    f"Incomplete download: got {downloaded / 1e6:.0f} MB "
                    f"of {total_size / 1e6:.0f} MB"
                )
            return
        except Exception as e:
            print(f"\n  Download interrupted: {e}")
            if attempt < max_retries:
                import time as _time
                wait = 5 * attempt
                print(f"  Retrying in {wait}s...")
                _time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to download {url} after {max_retries} attempts") from e


def extract_rar(rar_path, dest_dir):
    """Extract a RAR archive using unrar or 7z."""
    print(f"  Extracting {Path(rar_path).name}...")

    # Try unrar first
    try:
        subprocess.run(
            ["unrar", "x", "-o+", str(rar_path), str(dest_dir)],
            check=True, capture_output=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Try 7z (common on Windows)
    for cmd in ["7z", "7za", r"C:\Program Files\7-Zip\7z.exe"]:
        try:
            subprocess.run(
                [cmd, "x", str(rar_path), f"-o{dest_dir}", "-y"],
                check=True, capture_output=True,
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    # Try patoolib as fallback
    try:
        import patoolib
        patoolib.extract_archive(str(rar_path), outdir=str(dest_dir))
        return
    except ImportError:
        pass

    print("\n  ERROR: Cannot extract RAR file.")
    print("  Please install one of the following:")
    print("    - 7-Zip: https://www.7-zip.org/ (Windows)")
    print("    - unrar: sudo apt install unrar (Linux)")
    print("    - patoolib: pip install patool")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download UCF101 dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to download data to")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Download annotations first (small, reliable) ---
    splits_dir = data_dir / "ucfTrainTestlist"
    if splits_dir.exists() and any(splits_dir.iterdir()):
        print(f"[SKIP] Annotations directory already exists at {splits_dir}")
    else:
        print("=" * 50)
        print("Downloading Train/Test Split Annotations")
        print("=" * 50)

        zip_path = data_dir / "UCF101TrainTestSplits.zip"
        download_file(SPLITS_URL, zip_path)

        print(f"  Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        zip_path.unlink()
        print(f"  Annotations extracted to {splits_dir}")

    # --- Download and extract UCF101 videos (tries multiple mirrors) ---
    ucf_dir = data_dir / "UCF-101"
    if ucf_dir.exists() and any(ucf_dir.iterdir()):
        print(f"[SKIP] UCF-101 directory already exists at {ucf_dir}")
    else:
        print("\n" + "=" * 50)
        print("Downloading UCF101 Dataset (~6.5 GB)")
        print("This may take a while...")
        print("=" * 50)

        rar_path = data_dir / "UCF101.rar"
        for i, url in enumerate(UCF101_URLS):
            try:
                print(f"\n  Trying mirror {i+1}/{len(UCF101_URLS)}...")
                download_file(url, rar_path)
                break
            except Exception as e:
                print(f"  Mirror {i+1} failed: {e}")
                if rar_path.exists():
                    rar_path.unlink()
                if i == len(UCF101_URLS) - 1:
                    raise
        extract_rar(rar_path, data_dir)
        rar_path.unlink()
        print(f"  UCF101 extracted to {ucf_dir}")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)
    print(f"  Videos:      {ucf_dir}/")
    print(f"  Annotations: {splits_dir}/")

    if ucf_dir.exists():
        video_count = sum(1 for _ in ucf_dir.rglob("*.avi"))
        class_count = sum(1 for d in ucf_dir.iterdir() if d.is_dir())
        print(f"  Total videos: {video_count}")
        print(f"  Total classes: {class_count}")


if __name__ == "__main__":
    main()
