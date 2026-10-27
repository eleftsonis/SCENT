#!/usr/bin/env python3
"""Prepare the SCENT image set from public NYS.

Usage: python scripts/preprocess.py --src data/nys --out data/images [--workers 8]
(scripts/download.py runs this automatically after downloading.)
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SIZE = 256
QUALITY = 95


def convert(pair):
    src, dst = pair
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(dst, "JPEG", quality=QUALITY)


def preprocess(src, out, workers=8):
    pairs = [(p, out / p.parent.parent.name / "rgb" / (p.stem + ".jpg")) for p in sorted(src.glob("*/rgb/*_rgb.png"))]
    print(f"{len(pairs)} images -> {out}")
    with ProcessPoolExecutor(workers) as ex:
        list(tqdm(ex.map(convert, pairs, chunksize=16), total=len(pairs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root of the public NYS tree (session folders)")
    ap.add_argument("--out", required=True, help="output root for the 256x256 JPEGs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    preprocess(Path(args.src), Path(args.out), args.workers)
