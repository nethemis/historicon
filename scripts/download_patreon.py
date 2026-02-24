import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import feedparser
import requests

# Read Patreon RSS token from environment variable
PATREON_RSS_TOKEN = os.getenv("PATREON_RSS_TOKEN")
if not PATREON_RSS_TOKEN:
    print("❌ ERROR: PATREON_RSS_TOKEN environment variable is not set!")
    print("   Set it with: export PATREON_RSS_TOKEN='your_token_here'")
    sys.exit(1)

RSS_URL = f"https://www.patreon.com/rss/istorikon?auth={PATREON_RSS_TOKEN}&show=866770"
DOWNLOAD_DIR = "inputs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def sanitize_filename(filename):
    """
    Remove or replace problematic characters in filenames.
    - Replace spaces with underscores
    - Remove special characters except alphanumerics, underscores, hyphens, dots, and parentheses
    - Remove multiple consecutive underscores
    """
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")

    # Keep only alphanumerics, underscores, hyphens, dots, and parentheses
    # Remove other special characters
    filename = re.sub(r"[^\w\-\.\(\)]", "", filename)

    # Replace multiple consecutive underscores with a single one
    filename = re.sub(r"_+", "_", filename)

    # Remove leading/trailing underscores
    filename = filename.strip("_")

    return filename


def download_single_file(url, safe_name, path):
    """
    Download a single file from URL to path.
    Returns tuple: (status, safe_name, message, size_mb)
    """
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()

        downloaded_size = 0
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                downloaded_size += len(chunk)

        size_mb = downloaded_size / (1024 * 1024)
        return ("downloaded", safe_name, None, size_mb)
    except Exception as e:
        return ("error", safe_name, str(e), 0)


def download_patreon_audio():
    """
    Download audio files from Patreon RSS feed with sanitized filenames.
    Uses parallel downloads with ThreadPoolExecutor.
    """
    # Parse the feed
    feed = feedparser.parse(RSS_URL)
    print(f"Found {len(feed.entries)} entries in Patreon feed.\n")

    # Collect all download tasks
    download_tasks = []
    skipped = 0

    for entry in feed.entries:
        title = entry.get("title", "untitled").replace("/", "_")

        for link in entry.get("links", []):
            if link.get("rel") == "enclosure" and link.get("type", "").startswith(
                "audio/"
            ):
                url = link["href"]
                original_filename = os.path.basename(urlparse(url).path)

                # Get file extension
                _, ext = os.path.splitext(original_filename)

                # Create sanitized filename
                sanitized_title = sanitize_filename(title)
                safe_name = f"{sanitized_title}{ext}"
                path = os.path.join(DOWNLOAD_DIR, safe_name)

                # Check if a file with similar name already exists
                existing_files = [
                    f
                    for f in os.listdir(DOWNLOAD_DIR)
                    if f.startswith(sanitized_title) and f.endswith(ext)
                ]

                if existing_files:
                    print(f"⏭️  Skipping: {safe_name}")
                    print(f"   (found existing: {existing_files[0]})\n")
                    skipped += 1
                elif os.path.exists(path):
                    print(f"⏭️  Skipping: {safe_name} (already exists)\n")
                    skipped += 1
                else:
                    download_tasks.append((url, safe_name, path))

    # Download files in parallel
    downloaded = 0
    errors = 0

    if download_tasks:
        print(
            f"Starting parallel download of {len(download_tasks)} files (10 concurrent)...\n"
        )

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all download tasks
            future_to_file = {
                executor.submit(download_single_file, url, safe_name, path): safe_name
                for url, safe_name, path in download_tasks
            }

            # Process completed downloads as they finish
            for future in as_completed(future_to_file):
                status, safe_name, message, size_mb = future.result()

                if status == "downloaded":
                    print(f"✅ Downloaded: {safe_name} ({size_mb:.2f} MB)")
                    downloaded += 1
                elif status == "error":
                    print(f"❌ Error downloading {safe_name}: {message}")
                    errors += 1

    print("\n" + "=" * 70)
    print(f"Download complete!")
    print(f"Downloaded: {downloaded} files")
    print(f"Skipped: {skipped} files")
    if errors > 0:
        print(f"Errors: {errors} files")
    print(f"All files stored in: {DOWNLOAD_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    download_patreon_audio()
