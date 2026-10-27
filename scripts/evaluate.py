#!/usr/bin/env python3
"""Evaluate a SCENT checkpoint on the validation split.

Usage: python scripts/evaluate.py --config configs/scent.json --checkpoint weights.pt [--out metrics.json]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scent.data import NYSDataset, load_annotations, load_split, make_loader  # noqa: E402
from scent.evaluate import embed_dataset, format_row, retrieval_metrics, summary_row  # noqa: E402
from scent.model import SCENT  # noqa: E402

FROZEN_PREFIXES = ("clip.text_model.", "clip.text_projection.")


def build_model(cfg, checkpoint, device):
    model = SCENT(cfg["temperature"], cfg["lambda_is"], cfg["lambda_st"], cfg.get("clip"))
    if checkpoint:
        missing, unexpected = model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
        missing = [k for k in missing if not k.startswith(FROZEN_PREFIXES)]
        assert not missing and not unexpected, (missing, unexpected)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scent.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", help="write all metrics as JSON here")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = load_annotations(cfg["annotations"])
    ds = NYSDataset(cfg["image_root"], cfg["raw_root"], ann, load_split(cfg[f"{args.split}_split"]), train=False)
    loader = make_loader(ds, cfg["eval_batch_size"], False, cfg["workers"])
    model = build_model(cfg, args.checkpoint, device)
    z, _ = embed_dataset(model, loader, device)
    metrics = retrieval_metrics(z)
    print(format_row(summary_row(metrics)))
    for k in ("S2I", "I2S", "S2T", "T2S"):
        print(f"  {k:<4} " + " ".join(f"R@{n}={metrics[k][f'r{n}']:.1f}" for n in (1, 5, 10, 20)))
    if args.out:
        json.dump(metrics, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
