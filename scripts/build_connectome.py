"""Join the BANC edgelist onto the anatomy and write the CSR wiring.

    uv run python scripts/build_connectome.py [--min-synapses 5]

Input:  data/banc_anatomy.npz            (scripts/build_anatomy.py)
        data/banc/banc_888_edgelist_simple_v2.feather
Output: data/banc_v888.npz   CSR by presynaptic neuron, weights = sign x synapses

Edges whose endpoints are not in the kept neuron set are dropped: the anatomy
keeps proofread neurons with a super_class, the edgelist covers every segment.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("connectome")

EDGES = Path("data/banc/banc_888_edgelist_simple_v2.feather")
ANATOMY = Path("data/banc_anatomy.npz")


def to_index(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Position of each value in the sorted id table, -1 when absent."""
    order = np.argsort(ids)
    sorted_ids = ids[order]
    pos = np.searchsorted(sorted_ids, values)
    pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
    found = sorted_ids[pos_clipped] == values
    out = np.where(found, order[pos_clipped], -1)
    return out.astype(np.int64)


def build_csr(
    pre: np.ndarray, post: np.ndarray, weight: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows are presynaptic neurons, so one spike gathers one contiguous slice."""
    order = np.lexsort((post, pre))
    pre, post, weight = pre[order], post[order], weight[order]
    counts = np.bincount(pre, minlength=n)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, post.astype(np.int32), weight.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=Path, default=EDGES)
    ap.add_argument("--anatomy", type=Path, default=ANATOMY)
    ap.add_argument("--out", type=Path, default=Path("data/banc_v888.npz"))
    ap.add_argument("--min-synapses", type=int, default=5)
    ap.add_argument(
        "--keep-autapses",
        action="store_true",
        help="keep self-connections (dropped by default: see the log line below)",
    )
    args = ap.parse_args()

    with np.load(args.anatomy, allow_pickle=False) as f:
        ids, sign = f["ids"], f["sign"]
    log.info("anatomy: %d neurons", len(ids))

    edges = pl.read_ipc(args.edges, columns=["pre", "post", "count"]).with_columns(
        pl.col("pre").cast(pl.Int64), pl.col("post").cast(pl.Int64)
    )
    log.info(
        "edgelist: %d pairs, synapse count min %d max %d",
        len(edges),
        edges["count"].min(),
        edges["count"].max(),
    )

    strong = edges.filter(pl.col("count") >= args.min_synapses)
    log.info(
        "min_synapses=%d keeps %d pairs (%.1f%%)",
        args.min_synapses,
        len(strong),
        100 * len(strong) / len(edges),
    )

    pre = to_index(ids, strong["pre"].to_numpy())
    post = to_index(ids, strong["post"].to_numpy())
    known = (pre >= 0) & (post >= 0)
    log.info(
        "both endpoints known for %d pairs (%.1f%%); %d dropped",
        int(known.sum()),
        100 * float(known.mean()),
        int((~known).sum()),
    )
    pre, post = pre[known], post[known]
    counts = strong["count"].to_numpy()[known].astype(np.float32)

    self_edges = pre == post
    if self_edges.any():
        # Self-connections in this table are larger than real ones (mean 15.8 vs
        # 12.1 synapses) and for 11k neurons they carry over half the outgoing
        # weight -- segmentation artefacts, and at 0.275 mV per synapse they would
        # let a neuron re-excite itself past threshold on its own spike.
        log.info(
            "%s %d autapses (mean %.1f synapses vs %.1f for the rest)",
            "keeping" if args.keep_autapses else "dropping",
            int(self_edges.sum()),
            float(counts[self_edges].mean()),
            float(counts[~self_edges].mean()),
        )
        if not args.keep_autapses:
            pre, post, counts = pre[~self_edges], post[~self_edges], counts[~self_edges]

    weights = counts * sign[pre].astype(np.float32)
    indptr, indices, values = build_csr(pre, post, weights, len(ids))

    inhibitory = float((values < 0).mean())
    log.info(
        "CSR: %d edges, %.1f%% inhibitory, mean |weight| %.2f, densest row %d edges",
        len(values),
        100 * inhibitory,
        float(np.abs(values).mean()),
        int(np.diff(indptr).max()),
    )
    np.savez_compressed(
        args.out,
        indptr=indptr,
        indices=indices,
        weights=values,
        ids=ids,
        min_synapses=np.int32(args.min_synapses),
        autapses=np.bool_(args.keep_autapses),
    )
    log.info("wrote %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
