#!/usr/bin/env python3
"""Linear / MLP probing of the frozen smell representation z_S (appendix protocol).

Trains a probe on the train split and reports validation top-1 accuracy for the closed-set
`item` (43 classes) and `environment` (7 classes) labels of the annotation file.
Usage: python scripts/probe.py --checkpoint weights.pt [--config configs/scent.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scent.data import NYSDataset, load_annotations, load_split, make_loader  # noqa: E402
from scent.evaluate import embed_dataset  # noqa: E402
from scripts.evaluate import build_model  # noqa: E402


def fit_probe(x_tr, y_tr, x_va, y_va, n_cls, hidden, seed, epochs=300, lr=1e-3, wd=1e-4, bs=256):
    torch.manual_seed(seed)
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True) + 1e-6
    x_tr, x_va = (x_tr - mu) / sd, (x_va - mu) / sd
    net = nn.Linear(x_tr.shape[1], n_cls) if not hidden else nn.Sequential(
        nn.Linear(x_tr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_cls))
    net = net.to(x_tr.device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(x_tr), device=x_tr.device).split(bs):
            opt.zero_grad()
            F.cross_entropy(net(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return (net(x_va).argmax(1) == y_va).float().mean().item() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scent.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = load_annotations(cfg["annotations"])
    model = build_model(cfg, args.checkpoint, device)
    feats = {}
    for split in ("train", "val"):
        ds = NYSDataset(cfg["image_root"], cfg["raw_root"], ann, load_split(cfg[f"{split}_split"]), train=False)
        z, ids = embed_dataset(model, make_loader(ds, cfg["eval_batch_size"], False, cfg["workers"]), device)
        feats[split] = (z["z_s"], ids)
    for label in ("item", "environment"):
        classes = sorted({ann[i][label] for i in feats["train"][1]})
        cid = {c: k for k, c in enumerate(classes)}
        x, y = {}, {}
        for split in ("train", "val"):
            keep = [k for k, i in enumerate(feats[split][1]) if ann[i][label] in cid]
            x[split] = feats[split][0][keep]
            y[split] = torch.tensor([cid[ann[feats[split][1][k]][label]] for k in keep], device=device)
        lin = np.mean([fit_probe(x["train"], y["train"], x["val"], y["val"], len(classes), 0, s) for s in range(args.seeds)])
        mlp = np.mean([fit_probe(x["train"], y["train"], x["val"], y["val"], len(classes), 512, s) for s in range(args.seeds)])
        print(f"{label:<12} ({len(classes)} classes, chance {100 / len(classes):.1f}%): linear {lin:.1f}%  mlp {mlp:.1f}%")


if __name__ == "__main__":
    main()
