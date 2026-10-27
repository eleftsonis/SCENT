#!/usr/bin/env python3
"""Download the public New York Smells dataset and prepare it for SCENT.

1. Fetches the complete per-session tree from the Hugging Face mirror (cvlab/new-york-smells, ~45 GB)
   into <root>/<session>/..., or skips the download if --root already holds an extracted copy of the
   official archive (https://smell.cs.columbia.edu/static/smell-dataset.tar.gz, same layout under dataset/).
2. Converts every <session>/rgb/<i>_rgb.png to the JPEG the model is trained and evaluated on, under
   <out>/<session>/rgb/<i>_rgb.jpg (scripts/preprocess.py). The e-nose arrays are read from <root> unchanged.

Usage: python scripts/download.py --root data/nys --out data/images
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import preprocess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/nys", help="where the dataset tree goes (or already is)")
ap.add_argument("--out", default="data/images", help="output root for the prepared images")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

if any(Path(args.root).glob("*/rgb/*_rgb.png")):
    print(f"{args.root} already contains the dataset tree; skipping download")
else:
    snapshot_download("cvlab/new-york-smells", repo_type="dataset", local_dir=args.root, max_workers=args.workers)
preprocess(Path(args.root), Path(args.out), args.workers)
