#!/usr/bin/env python3
"""Train SCENT.  Usage: python scripts/train.py --config configs/scent.json [--output_dir runs/x]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scent.train import train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/scent.json")
ap.add_argument("--output_dir")
ap.add_argument("--epochs", type=int)
ap.add_argument("--wandb_project")
args = ap.parse_args()
cfg = json.load(open(args.config))
cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
train(cfg)
