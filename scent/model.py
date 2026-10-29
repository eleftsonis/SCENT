"""SCENT model: smell transformer with two projection heads aligned to CLIP image and text spaces."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from transformers.utils import logging as hf_logging

CLIP_NAME = "openai/clip-vit-base-patch16"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the tokenizer runs in the main process before DataLoader workers fork
hf_logging.set_verbosity_error()  # transformers >= 5 otherwise reports CLIP's unused position_ids buffers on every load
hf_logging.disable_progress_bar()


class SmellEncoder(nn.Module):
    """(B, 28, 32) raw e-nose signal -> (B, 768) representation z_S."""

    def __init__(self, n_steps=28, n_sensors=32, dim=448, out_dim=768, layers=6, heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_sensors, dim)
        self.pos = nn.Parameter(torch.randn(1, n_steps, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
        self.output_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x) + self.pos
        return self.output_proj(self.transformer(x).mean(dim=1))


def _features(out):
    """CLIP feature functions return a tensor (transformers < 5.x) or a model output (>= 5.x)."""
    return out if torch.is_tensor(out) else out.pooler_output


def projection_head(in_dim=768, out_dim=512):
    return nn.Sequential(nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim, eps=1e-12), nn.GELU(),
                         nn.Linear(in_dim, out_dim))


class SCENT(nn.Module):
    def __init__(self, temperature=0.07, lambda_is=0.5, lambda_st=0.5, clip_name=None):
        """clip_name: Hugging Face id or a local directory with the CLIP weights (offline clusters)."""
        super().__init__()
        clip_name = clip_name or CLIP_NAME
        self.smell_encoder = SmellEncoder()
        self.head_s2i = projection_head()
        self.head_s2t = projection_head()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.lambda_is, self.lambda_st = lambda_is, lambda_st
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        for p in self.clip.text_model.parameters():
            p.requires_grad_(False)
        self.clip.text_projection.requires_grad_(False)

    # --- encoders -------------------------------------------------------
    def encode_smell(self, smell):
        """Returns z_S (768-d), z_S->I and z_S->T (512-d, L2-normalised)."""
        z = self.smell_encoder(smell)
        return z, F.normalize(self.head_s2i(z), dim=-1), F.normalize(self.head_s2t(z), dim=-1)

    def encode_image(self, pixels):
        return F.normalize(_features(self.clip.get_image_features(pixel_values=pixels)), dim=-1)

    @torch.no_grad()
    def encode_text(self, captions):
        tok = self.tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        tok = {k: v.to(self.temperature.device) for k, v in tok.items()}
        return F.normalize(_features(self.clip.get_text_features(**tok)), dim=-1)

    # --- training -------------------------------------------------------
    def loss(self, batch):
        z_i = self.encode_image(batch["image"])
        _, z_s2i, z_s2t = self.encode_smell(batch["smell"])
        z_t = self.encode_text(batch["caption"]).to(z_i.dtype)
        loss_is = self._symmetric_infonce(z_s2i, z_i)
        loss_st = self._symmetric_infonce(z_s2t, z_t)
        total = self.lambda_is * loss_is + self.lambda_st * loss_st
        return total, {"loss": total.item(), "loss_is": loss_is.item(), "loss_st": loss_st.item(),
                       "temperature": self.temperature.item()}

    def _symmetric_infonce(self, a, b):
        targets = torch.arange(len(a), device=a.device)
        logits_ab = a @ b.detach().t() / self.temperature
        logits_ba = b @ a.detach().t() / self.temperature
        return (F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets)) / 2

    # --- inference ------------------------------------------------------
    @torch.no_grad()
    def embed(self, batch):
        z_s, z_s2i, z_s2t = self.encode_smell(batch["smell"])
        return {"z_s": z_s, "z_s2i": z_s2i, "z_s2t": z_s2t, "z_i": self.encode_image(batch["image"]),
                "z_t": self.encode_text(batch["caption"])}
