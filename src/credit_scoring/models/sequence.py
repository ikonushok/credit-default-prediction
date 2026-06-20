"""Sequence model over a client's credit-product history (phase 3).

Each `id` is a variable-length sequence of products (ordered by `rn`), each
product a vector of 59 integer-encoded columns. **All 59 columns are categorical
codes** (bins 0..19, statuses 0..4, flags) so each is given its own embedding
(implemented with a single shared table + per-column offsets), concatenated per
timestep and read by a bidirectional GRU; the final state feeds an MLP head ->
default logit. This is the canonical strong approach for this dataset and
replaces the earlier "scaled-numeric GRU".

Storage is compact: all rows are kept once as an int16 matrix plus per-id
(offset, length); batches pad dynamically. Padded timesteps are ignored by
`pack_padded_sequence`. Folds use the SAME id-level make_folds(seed) as the GBDT
runs, so OOF/test predictions align by id and are comparable/ensemble-able.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .. import config as C
from .. import metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SeqStore:
    """Compact per-id sequence store of integer category codes (ascending id)."""

    def __init__(self, df, feature_cols: list[str]):
        self.feature_cols = feature_cols
        # Each id's rows are contiguous; np.unique gives ascending uniq ids with
        # first-occurrence offsets and counts indexing the contiguous block.
        ids = df[C.ID_COL].to_numpy()
        self.uniq_ids, starts, counts = np.unique(ids, return_index=True, return_counts=True)
        self.uniq_ids = self.uniq_ids.astype("int32")
        self.offsets = starts.astype("int64")
        self.lengths = counts.astype("int32")
        self.values = df[feature_cols].to_numpy("int16")  # codes, (n_rows, F)


def cardinalities(train_store: SeqStore, test_store: SeqStore) -> np.ndarray:
    """Per-column vocabulary size = max code over train+test + 1."""
    tr = train_store.values.max(axis=0)
    te = test_store.values.max(axis=0)
    return (np.maximum(tr, te) + 1).astype("int64")


def _make_batch(store: SeqStore, idx: np.ndarray, max_len: int):
    """Padded (B, L, F) int64 code tensor + lengths for a batch of id-positions."""
    lens = np.minimum(store.lengths[idx], max_len)
    L = int(lens.max())
    F = store.values.shape[1]
    x = np.zeros((len(idx), L, F), dtype="int64")
    for b, i in enumerate(idx):
        ln = lens[b]
        s = store.offsets[i] + store.lengths[i] - ln   # keep most recent products
        x[b, :ln] = store.values[s:s + ln]
    return torch.from_numpy(x), torch.from_numpy(lens.astype("int64"))


class EmbGRUClassifier(nn.Module):
    def __init__(self, cards: np.ndarray, emb_dim: int = 8, hidden: int = 128,
                 layers: int = 1, dropout: float = 0.1, bidirectional: bool = True,
                 pooling: str = "last"):
        super().__init__()
        cards = [int(c) for c in cards]
        offsets = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype("int64")
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embed = nn.Embedding(int(sum(cards)), emb_dim)
        nn.init.normal_(self.embed.weight, std=0.05)
        F = len(cards)
        self.input_dim = F * emb_dim
        self.gru = nn.GRU(self.input_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0,
                          bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.pooling = pooling
        if pooling == "attention":
            self.attn = nn.Linear(out_dim, 1)
        # attention pooling concatenates pooled + last hidden -> 2*out_dim head input
        head_in = out_dim * 2 if pooling == "attention" else out_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.bidirectional = bidirectional

    def _last_hidden(self, h):
        return torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]

    def _head_input(self, x, lengths):                   # x: (B, L, F) long
        e = self.embed(x + self.offsets)                 # (B, L, F, D)
        e = e.flatten(start_dim=2)                       # (B, L, F*D)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h = self.gru(packed)
        last = self._last_hidden(h)                      # (B, out_dim)
        if self.pooling != "attention":
            return last
        # attention pooling over all valid timesteps, concatenated with last hidden
        out, _ = pad_packed_sequence(out_packed, batch_first=True)   # (B, L, out_dim)
        L = out.size(1)
        mask = torch.arange(L, device=out.device)[None, :] < lengths.to(out.device)[:, None]
        scores = self.attn(out).squeeze(-1)              # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)   # (B, L, 1)
        pooled = (w * out).sum(dim=1)                    # (B, out_dim)
        return torch.cat([pooled, last], dim=-1)

    def forward(self, x, lengths):                       # x: (B, L, F) long
        return self.head(self._head_input(x, lengths)).squeeze(-1)

    def forward_repr(self, x, lengths):
        """Penultimate `hidden`-d representation ReLU(head[0](pooled)) for stacking."""
        return torch.relu(self.head[0](self._head_input(x, lengths)))


class EmbTransformerClassifier(nn.Module):
    """Same per-column embedding front-end as the GRU, but a self-attention
    Transformer encoder over timesteps (different inductive bias → decorrelates
    from the recurrent model for ensembling). Learned positional embeddings make
    it order-aware; padded steps are masked out of attention and pooling."""

    def __init__(self, cards: np.ndarray, emb_dim: int = 8, d_model: int = 128,
                 nhead: int = 4, layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.1, hidden: int = 128, max_len: int = 40):
        super().__init__()
        cards = [int(c) for c in cards]
        offsets = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype("int64")
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embed = nn.Embedding(int(sum(cards)), emb_dim)
        nn.init.normal_(self.embed.weight, std=0.05)
        F = len(cards)
        self.proj = nn.Linear(F * emb_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos.weight, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.attn = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.max_len = max_len

    def forward(self, x, lengths):                       # x: (B, L, F) long
        B, L, _ = x.shape
        e = self.embed(x + self.offsets).flatten(start_dim=2)   # (B, L, F*D)
        h = self.proj(e)                                 # (B, L, d_model)
        pos_ids = torch.arange(L, device=x.device).clamp_max(self.max_len - 1)
        h = h + self.pos(pos_ids)[None]                  # broadcast positions
        pad_mask = torch.arange(L, device=x.device)[None, :] >= lengths.to(x.device)[:, None]
        h = self.encoder(h, src_key_padding_mask=pad_mask)       # (B, L, d_model)
        scores = self.attn(h).squeeze(-1).masked_fill(pad_mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)   # (B, L, 1)
        pooled = (w * h).sum(dim=1)                       # (B, d_model)
        return self.head(pooled).squeeze(-1)


def _predict(model, store, idx, device, max_len, batch_size):
    model.eval()
    out = np.zeros(len(idx), dtype="float32")
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            x, lens = _make_batch(store, b, max_len)
            logit = model(x.to(device), lens.to(device))
            out[s:s + batch_size] = torch.sigmoid(logit).float().cpu().numpy()
    return out


def _predict_repr(model, store, idx, device, max_len, batch_size, repr_dim):
    """Penultimate representation per id (float16), for embedding-stacking."""
    model.eval()
    out = np.zeros((len(idx), repr_dim), dtype="float16")
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            x, lens = _make_batch(store, b, max_len)
            r = model.forward_repr(x.to(device), lens.to(device))
            out[s:s + batch_size] = r.float().cpu().numpy().astype("float16")
    return out


def train_cv(train_store, y, folds, test_store, params=None, seed: int = 42,
             collect_repr: bool = False):
    """Train one embedding bi-GRU per fold. Returns (oof, test_pred, per_fold_auc).

    If ``collect_repr`` (gru arch only), also returns ``(oof_emb, test_emb)``:
    leak-safe penultimate representations per id — each fold's val/test rows come
    from a model that did not train on them — for embedding-stacking into a GBDT.
    """
    p = {"arch": "gru", "emb_dim": 8, "hidden": 128, "layers": 1, "dropout": 0.1,
         "bidirectional": True, "pooling": "last", "lr": 1e-3, "epochs": 8,
         "batch_size": 2048, "max_len": 40, "patience": 2,
         # transformer-only knobs (ignored by gru)
         "d_model": 128, "nhead": 4, "dim_feedforward": 256}
    if params:
        p.update(params)
    device = get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cards = cardinalities(train_store, test_store)
    pos_weight = torch.tensor(
        [(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32, device=device)
    n_folds = int(folds.max()) + 1
    oof = np.zeros(len(y), dtype="float32")
    test_pred = np.zeros(len(test_store.uniq_ids), dtype="float32")
    test_idx = np.arange(len(test_store.uniq_ids))
    per_fold = []
    if collect_repr:
        repr_dim = int(p["hidden"])
        oof_emb = np.zeros((len(y), repr_dim), dtype="float16")
        test_emb = np.zeros((len(test_store.uniq_ids), repr_dim), dtype="float32")

    for k in range(n_folds):
        tr_idx = np.where(folds != k)[0]
        va_idx = np.where(folds == k)[0]
        if p["arch"] == "transformer":
            model = EmbTransformerClassifier(
                cards, p["emb_dim"], p["d_model"], p["nhead"], p["layers"],
                p["dim_feedforward"], p["dropout"], p["hidden"], p["max_len"]).to(device)
        else:
            model = EmbGRUClassifier(cards, p["emb_dim"], p["hidden"], p["layers"],
                                     p["dropout"], p["bidirectional"], p["pooling"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc, best_state, bad = -1.0, None, 0
        for epoch in range(p["epochs"]):
            model.train()
            perm = np.random.permutation(tr_idx)
            for s in range(0, len(perm), p["batch_size"]):
                b = perm[s:s + p["batch_size"]]
                x, lens = _make_batch(train_store, b, p["max_len"])
                yb = torch.from_numpy(y[b].astype("float32")).to(device)
                opt.zero_grad()
                loss = loss_fn(model(x.to(device), lens.to(device)), yb)
                loss.backward()
                opt.step()
            va_pred = _predict(model, train_store, va_idx, device, p["max_len"], p["batch_size"])
            auc = metrics.roc_auc(y[va_idx], va_pred)
            print(f"[seq] fold {k} epoch {epoch}: val_auc={auc:.5f}", flush=True)
            if auc > best_auc + 1e-5:
                best_auc, bad = auc, 0
                best_state = {kk: v.cpu().clone() for kk, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= p["patience"]:
                    break

        model.load_state_dict(best_state)
        oof[va_idx] = _predict(model, train_store, va_idx, device, p["max_len"], p["batch_size"])
        test_pred += _predict(model, test_store, test_idx, device, p["max_len"], p["batch_size"]) / n_folds
        if collect_repr:
            oof_emb[va_idx] = _predict_repr(
                model, train_store, va_idx, device, p["max_len"], p["batch_size"], repr_dim)
            test_emb += _predict_repr(
                model, test_store, test_idx, device, p["max_len"], p["batch_size"], repr_dim
            ).astype("float32") / n_folds
        per_fold.append(best_auc)
        print(f"[seq] fold {k}: best val_auc={best_auc:.5f}", flush=True)

    if collect_repr:
        return oof, test_pred, per_fold, oof_emb, test_emb
    return oof, test_pred, per_fold
