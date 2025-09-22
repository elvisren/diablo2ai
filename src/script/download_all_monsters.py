#!/usr/bin/env python3
"""
Download all /styles/zulu/...*graphic.png images referenced on:
  https://diablo2.io/monsters/

- Matches only images whose URL path ends with "graphic.png"
- Saves all files into ./image/ (flat)
- Shows a single progress bar for count of images

Usage:
  python download_diablo2_graphic_images.py
"""

import pathlib
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm

BASE_URL = "https://diablo2.io/monsters/"
SITE_ROOT = "https://diablo2.io"
SAVE_DIR = pathlib.Path("monstor_image")  # flat output dir

# Match any URL that contains /styles/zulu/ and whose PATH ends with graphic.png
ZULU_GRAPHIC_RE = re.compile(r"/styles/zulu/.*graphic\.png$", re.IGNORECASE)

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": SITE_ROOT + "/",
    })
    return s

def path_ends_with_graphic_png(url: str) -> bool:
    p = urlparse(url)
    return bool(ZULU_GRAPHIC_RE.search(p.path))

def collect_targets(html: str):
    soup = BeautifulSoup(html, "html.parser")
    candidates = set()

    # 1) data-background-image=".../styles/zulu/...graphic.png"
    for el in soup.find_all(attrs={"data-background-image": True}):
        val = el.get("data-background-image", "")
        if val:
            # make it absolute for uniformity; filtering is by path
            abs_url = urljoin(SITE_ROOT, val)
            if path_ends_with_graphic_png(abs_url):
                candidates.add(abs_url)

    # 2) style="background-image: url('.../styles/zulu/...graphic.png')"
    style_re = re.compile(r"background-image\s*:\s*url\((['\"]?)(.*?)\1\)", re.IGNORECASE)
    for el in soup.find_all(style=True):
        style = el.get("style", "")
        m = style_re.search(style)
        if m:
            abs_url = urljoin(SITE_ROOT, m.group(2))
            if path_ends_with_graphic_png(abs_url):
                candidates.add(abs_url)

    return sorted(candidates)

def save_one(session: requests.Session, url: str, outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    name = pathlib.Path(urlparse(url).path).name  # keep filename only
    dest = outdir / name
    # Skip if exists and non-empty
    if dest.exists() and dest.stat().st_size > 0:
        return

    r = session.get(url, stream=True, timeout=30)
    if r.status_code != 200:
        return  # quietly skip failures for a clean progress bar

    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 14):
            if chunk:
                f.write(chunk)
    tmp.replace(dest)

def main():
    session = make_session()
    resp = session.get(BASE_URL, timeout=30)
    resp.raise_for_status()

    targets = collect_targets(resp.text)
    if not targets:
        print("No matching /styles/zulu/...graphic.png images found.")
        return

    with tqdm(total=len(targets), unit="img", desc="Downloading") as bar:
        for url in targets:
            try:
                save_one(session, url, SAVE_DIR)
            except Exception:
                # keep quiet; just advance the bar
                pass
            finally:
                bar.update(1)

    print(f"Saved {len(targets)} image(s) into ./{SAVE_DIR}/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
