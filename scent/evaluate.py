"""Retrieval evaluation (S2I, I2S, S2T, T2S, S2IT) and the Table-2 summary row."""
import torch
import torch.nn.functional as F

KS = (1, 5, 10, 20)
ALPHAS = [i / 10 for i in range(10)]


def recall_at_k(scores):
    """scores: (N, N) query x gallery, ground truth on the diagonal. Returns {r1, r5, r10, r20, mean_rank}."""
    order = scores.argsort(dim=1, descending=True)
    rank = (order == torch.arange(len(scores), device=scores.device)[:, None]).float().argmax(1)
    out = {f"r{k}": round((rank < k).float().mean().item() * 100, 1) for k in KS}
    out["mean_rank"] = round(rank.float().mean().item() + 1, 1)
    return out


@torch.no_grad()
def embed_dataset(model, loader, device):
    """Concatenate model.embed() over a loader -> dict of (N, d) tensors (+ ids)."""
    model.eval()
    parts, ids = {}, []
    for batch in loader:
        ids += batch["id"]
        out = model.embed({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})
        for k, v in out.items():
            parts.setdefault(k, []).append(v.float())
    return {k: torch.cat(v) for k, v in parts.items()}, ids


def retrieval_metrics(z):
    """All retrieval tasks from embeddings. S2IT reports every alpha for both fusion types."""
    s2i = z["z_s2i"] @ z["z_i"].t()
    s2t = z["z_s2t"] @ z["z_t"].t()
    m = {"S2I": recall_at_k(s2i), "I2S": recall_at_k(s2i.t()), "S2T": recall_at_k(s2t), "T2S": recall_at_k(s2t.t())}
    for a in ALPHAS:
        b = 1 - a
        m[f"S2IT_sim_a{a:.1f}"] = recall_at_k(a * s2i + b * s2t)
        blend = F.normalize(a * z["z_s2i"] + b * z["z_s2t"], dim=-1)
        m[f"S2IT_latent_a{a:.1f}"] = recall_at_k(a * (blend @ z["z_i"].t()) + b * (blend @ z["z_t"].t()))
    return m


def best_alpha(metrics, fusion):
    keys = [k for k in metrics if k.startswith(f"S2IT_{fusion}_a")]
    return max(keys, key=lambda k: (metrics[k]["r5"], metrics[k]["r1"]))


def summary_row(metrics):
    """Table-2 row: S2I, S2T and the best S2IT (over fusion type and alpha), plus the joint score."""
    sim, lat = best_alpha(metrics, "sim"), best_alpha(metrics, "latent")
    s2it_key = max((sim, lat), key=lambda k: (metrics[k]["r5"], metrics[k]["r1"]))
    row = {"S2I": metrics["S2I"], "S2T": metrics["S2T"], "S2IT": metrics[s2it_key], "S2IT_setting": s2it_key[5:]}
    row["joint"] = round(row["S2I"]["r5"] + row["S2T"]["r5"] + row["S2IT"]["r5"], 1)
    return row


def format_row(row):
    rec = lambda d: "/".join(f"{d[f'r{k}']:.1f}" for k in KS)
    return (f"S2I {rec(row['S2I'])} | S2T {rec(row['S2T'])} | S2IT {rec(row['S2IT'])} ({row['S2IT_setting']}) "
            f"| joint {row['joint']:.1f}")
