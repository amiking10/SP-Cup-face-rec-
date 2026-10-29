"""Download VPCD metadata or selected ChokePoint archives to data/raw/.

No dataset files are committed to Git. This script avoids downloading full 12 GB
ChokePoint by default and verifies MD5 when the record supplies it.

VPCD: supply a direct metadata archive URL obtained from the *official* website.
Its video footage is not part of the annotations and may require separate rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CHOKEPOINT_RECORD_API = "https://zenodo.org/api/records/815657"
CHOKEPOINT_STARTER = ("groundtruth.tar.xz", "P2E_S5.tar.xz")
USER_AGENT = "SPCup-C3-Dataset-Downloader/1.0"


def request(url):
    return Request(url, headers={"User-Agent": USER_AGENT})


def fetch_json(url):
    with urlopen(request(url), timeout=45) as response:
        return json.load(response)


def download(url: str, destination: Path, expected_md5: str | None = None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_md5 and md5_file(destination) != expected_md5:
            raise ValueError(f"existing file checksum mismatch: {destination}")
        print(f"Already present: {destination}")
        return
    partial = destination.with_name(destination.name + ".part")
    try:
        with urlopen(request(url), timeout=60) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if expected_md5 and md5_file(partial) != expected_md5:
            raise ValueError(f"downloaded file checksum mismatch: {destination}")
        os.replace(partial, destination)
        print(f"Saved: {destination}")
    finally:
        # Keep partial downloads for inspection / possible manual resume.
        pass


def md5_file(path: Path):
    h = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def chokepoint(out: Path, files: list[str], list_only: bool):
    record = fetch_json(CHOKEPOINT_RECORD_API)
    entries = {f["key"]: f for f in record["files"]}
    print("Available:", ", ".join(sorted(entries)))
    if list_only:
        return
    for name in files:
        if name not in entries:
            raise ValueError(f"not in Zenodo record: {name}")
        file = entries[name]
        checksum = file.get("checksum", "")
        expected = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        link = file.get("links", {}).get("self") or file.get("links", {}).get("content")
        if not link:
            raise ValueError(f"no downloadable link for {name}")
        download(link, out / name, expected)


def vpcd(out: Path, url: str | None):
    if not url:
        raise ValueError(
            "VPCD requires its direct metadata archive URL from "
            "https://www.robots.ox.ac.uk/~vgg/data/Video_Person_Clustering/ "
            "(click 'All Annotations, Features and Labels'). "
            "Paste that URL using --vpcd-url; no movie footage will be downloaded."
        )
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname is None:
        raise ValueError("VPCD archive URL must be HTTPS")
    filename = Path(parsed.path).name or "vpcd_annotations.tar.gz"
    if not filename.endswith((".tar.gz", ".tgz", ".zip")):
        raise ValueError("expected VPCD metadata archive .tar.gz/.tgz/.zip")
    download(url, out / filename)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=["chokepoint", "vpcd"])
    p.add_argument("--output", type=Path, help="Override local dataset output directory")
    p.add_argument("--list", action="store_true", help="List available ChokePoint files")
    p.add_argument("--files", nargs="+", default=list(CHOKEPOINT_STARTER),
                   help="Zenodo ChokePoint file keys; defaults to small gt and crowded sequence")
    p.add_argument("--vpcd-url", help="Direct VPCD annotations archive URL from official Oxford site")
    args = p.parse_args()
    folder = args.output or Path("datasets/data/raw") / args.dataset
    if args.dataset == "chokepoint":
        chokepoint(folder, args.files, args.list)
    else:
        vpcd(folder, args.vpcd_url)


if __name__ == "__main__":
    main()
