"""NYS dataset for SCENT: images, raw e-nose signals and VLM captions."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
STAGE_STEPS = 14  # timesteps kept per stage (baseline, sample)


def load_annotations(path):
    """annotations.jsonl -> {id: record}."""
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def load_split(path):
    with open(path) as f:
        return json.load(f)["ids"]


def fit_stage(x):
    """Truncate to STAGE_STEPS rows, or repeat the last row to reach them."""
    if len(x) >= STAGE_STEPS:
        return x[:STAGE_STEPS]
    return np.concatenate([x, np.repeat(x[-1:], STAGE_STEPS - len(x), axis=0)])


def load_smell(raw_root, sample_id):
    """(28, 32) float32: baseline stage followed by sample stage, raw sensor values."""
    session, idx = sample_id.rsplit("_", 1)
    base = np.load(Path(raw_root) / session / "baseline_raw" / f"{idx}_baseline.npy")
    samp = np.load(Path(raw_root) / session / "sample_raw" / f"{idx}_sample.npy")
    return torch.from_numpy(np.concatenate([fit_stage(base), fit_stage(samp)]).astype(np.float32))


def caption(rec):
    """VLM caption used for training and as the text gallery at evaluation: object, context, smells."""
    smells = ", ".join(rec["smells"])
    base = f"{rec['object']} in {rec['background']}"
    return f"{base} with smells of {smells}" if smells else base


def image_transform(train):
    """Training augmentations run on the PIL image (uint8) before ToTensor: same torchvision transforms and
    parameter sampling as on float tensors, at a third of the CPU cost per sample."""
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.15),
            T.RandomRotation(15),
            T.RandomAffine(0, translate=(0.15, 0.15)),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    return T.Compose([T.ToTensor(), T.Resize((224, 224)), T.Normalize(CLIP_MEAN, CLIP_STD)])


class NYSDataset(Dataset):
    """One item: CLIP-ready image, (28, 32) smell tensor, caption, id.

    The e-nose arrays of the whole split are loaded into memory once (a few MB), so each item only
    decodes one JPEG; this matters on networked filesystems.
    """

    def __init__(self, image_root, raw_root, annotations, split_ids, train):
        self.image_root = Path(image_root)
        self.ann = annotations
        self.ids = [i for i in split_ids if i in annotations]
        self.smells = {i: load_smell(raw_root, i) for i in self.ids}
        self.train = train
        self.tf = image_transform(train)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        session, idx = sid.rsplit("_", 1)
        with Image.open(self.image_root / session / "rgb" / f"{idx}_rgb.jpg") as im:
            image = self.tf(im.convert("RGB"))
        rec = self.ann[sid]
        return {"id": sid, "image": image, "smell": self.smells[sid], "caption": caption(rec)}


def collate(items):
    out = {k: [it[k] for it in items] for k in items[0]}
    out["image"] = torch.stack(out["image"])
    out["smell"] = torch.stack(out["smell"])
    return out


def seed_worker(_):
    torch.set_num_threads(1)  # one thread per loader process; avoids CPU oversubscription with many workers
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def make_loader(dataset, batch_size, train, workers, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=workers,
                      collate_fn=collate, pin_memory=True, persistent_workers=workers > 0, generator=g,
                      worker_init_fn=seed_worker, prefetch_factor=4 if workers else None)
