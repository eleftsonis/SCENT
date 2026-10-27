"""Training loop: AdamW, warmup-stable-decay schedule, fp16 autocast, EMA weights, periodic evaluation."""
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import NYSDataset, load_annotations, load_split, make_loader
from .evaluate import embed_dataset, format_row, retrieval_metrics, summary_row
from .model import SCENT

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_factor(step, total, warmup, cooldown):
    """Linear warmup, constant peak, linear cooldown to zero over the last `cooldown` fraction."""
    x = step / total
    if x < warmup:
        return x / warmup
    if x < 1 - cooldown:
        return 1.0
    return max(0.0, (1 - x) / cooldown)


class EMA:
    """Exponential moving average of all floating-point tensors in the model state."""

    def __init__(self, model, decay):
        self.decay, self.updates = decay, 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.float(), alpha=1 - d)

    def state_dict(self, model):
        return {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v) for k, v in model.state_dict().items()}

    @torch.no_grad()
    def swap(self, model):
        """Load EMA weights into the model, returning a function that restores the raw weights."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(self.state_dict(model))

        def restore():
            sd = model.state_dict()
            for k, v in backup.items():
                sd[k].copy_(v)
        return restore


def param_groups(model, weight_decay):
    decay = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith("bias")]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith("bias")]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def save_weights(state, path):
    torch.save({k: v.cpu() for k, v in state.items()
                if not k.startswith(("clip.text_model.", "clip.text_projection."))}, path)


def evaluate(model, loader, device):
    z, _ = embed_dataset(model, loader, device)
    return summary_row(retrieval_metrics(z))


def train(cfg):
    set_seed(cfg["seed"])
    device = "cuda"
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    ann = load_annotations(cfg["annotations"])
    train_ds = NYSDataset(cfg["image_root"], cfg["raw_root"], ann, load_split(cfg["train_split"]), train=True)
    val_ds = NYSDataset(cfg["image_root"], cfg["raw_root"], ann, load_split(cfg["val_split"]), train=False)
    train_loader = make_loader(train_ds, cfg["batch_size"], True, cfg["workers"], cfg["seed"])
    val_loader = make_loader(val_ds, cfg["eval_batch_size"], False, cfg["workers"])

    model = SCENT(cfg["temperature"], cfg["lambda_is"], cfg["lambda_st"], cfg.get("clip", None)).to(device)
    if cfg.get("gradient_checkpointing"):
        model.clip.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                  eps=cfg.get("adam_eps", 1e-8))
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(model, cfg["ema_decay"])

    steps_per_epoch = len(train_loader)
    total = cfg["epochs"] * steps_per_epoch
    eval_every = max(1, total // cfg["evaluations"])
    print(f"train {len(train_ds)} | val {len(val_ds)} | {steps_per_epoch} steps/epoch | {total} steps | eval every {eval_every}")

    wandb = None
    if cfg.get("wandb_project"):
        import wandb
        wandb.init(project=cfg["wandb_project"], name=out.name, config=cfg)

    log_path = out / "log.csv"
    best = {"raw": -1.0, "ema": -1.0}
    step, t0 = 0, time.time()
    model.train()
    while step < total:
        for batch in train_loader:
            factor = lr_factor(step, total, cfg["warmup_fraction"], cfg["cooldown_fraction"])
            for g in optimizer.param_groups:
                g["lr"] = cfg["lr"] * factor
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss, logs = model.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            step += 1

            if step % 50 == 0:
                print(f"step {step}/{total} lr {cfg['lr'] * factor:.2e} loss {logs['loss']:.3f} "
                      f"(is {logs['loss_is']:.3f} st {logs['loss_st']:.3f}) tau {logs['temperature']:.4f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
                if wandb:
                    wandb.log({"train/" + k: v for k, v in logs.items()} | {"train/lr": cfg["lr"] * factor}, step=step)

            if step % eval_every == 0 or (step == total and total % eval_every):
                row_raw = evaluate(model, val_loader, device)
                restore = ema.swap(model)
                row_ema = evaluate(model, val_loader, device)
                restore()
                model.train()
                print(f"[eval step {step}] raw: {format_row(row_raw)}")
                print(f"[eval step {step}] ema: {format_row(row_ema)}", flush=True)
                log_eval(log_path, step, cfg["lr"] * factor, row_raw, row_ema, wandb)
                if row_raw["joint"] >= best["raw"]:
                    best["raw"] = row_raw["joint"]
                    save_weights(model.state_dict(), out / "best_raw.pt")
                if row_ema["joint"] >= best["ema"]:
                    best["ema"] = row_ema["joint"]
                    save_weights(ema.state_dict(model), out / "best_ema.pt")
                save_weights(ema.state_dict(model), out / "last_ema.pt")
            if step >= total:
                break
    print(f"done. best joint: raw {best['raw']:.1f}, ema {best['ema']:.1f}. weights in {out}")


def log_eval(path, step, lr, row_raw, row_ema, wandb):
    flat = {"step": step, "lr": f"{lr:.3e}"}
    for tag, row in (("raw", row_raw), ("ema", row_ema)):
        for task in ("S2I", "S2T", "S2IT"):
            for k in (1, 5, 10, 20):
                flat[f"{tag}_{task}_r{k}"] = row[task][f"r{k}"]
        flat[f"{tag}_S2IT_setting"] = row["S2IT_setting"]
        flat[f"{tag}_joint"] = row["joint"]
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat))
        if new:
            w.writeheader()
        w.writerow(flat)
    if wandb:
        wandb.log({"eval/" + k: v for k, v in flat.items() if isinstance(v, (int, float))}, step=step)
