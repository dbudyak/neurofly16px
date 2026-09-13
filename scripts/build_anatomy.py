"""Build data/banc_anatomy.npz and data/populations.json from the public BANC tables.

    uv run python scripts/build_anatomy.py

Needs only the files in `htem/BANC-project` (no login):
    data/banc/banc_888_meta.parquet
    data/banc/banc_neck_functional_classes.csv
Download them with scripts/fetch_banc.sh. The wiring is a separate, gated step
(docs/banc.md, "Confirm on the host").
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from neurofly16px.brain.anatomy import CATEGORICAL, transmitter_sign

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("anatomy")

META = Path("data/banc/banc_888_meta.parquet")
NECK = Path("data/banc/banc_neck_functional_classes.csv")

# Johnston's organ sub-classes: sound (A, B) and wind/gravity (C, E), docs/banc.md.
JO_SOUND = ("johnstons_organ_A_neuron", "johnstons_organ_B_neuron")
JO_WIND = ("johnstons_organ_C_neuron", "johnstons_organ_E_neuron")
NAMED_DNS = ("DNp01", "DNa02", "MDN", "DNp09", "DNa01", "DNb01", "DNg13")
SUPER_CLUSTERS = {
    "dn_walking": "walking",
    "dn_takeoff_landing": "takeoff-landing",
    "dn_threat_response": "threat response",
    "dn_flight_steering_1": "flight steering 1",
    "dn_flight_steering_2": "flight steering 2",
    "dn_head_orienting": "head orienting",
}


def parse_positions(column: pl.Series) -> np.ndarray:
    """'390720, 93568, 86220' -> float32 (n, 3) in nm; NaN where missing."""
    out = np.full((len(column), 3), np.nan, np.float32)
    for i, value in enumerate(column.to_list()):
        if not value:
            continue
        parts = value.replace("[", "").replace("]", "").split(",")
        if len(parts) != 3:
            continue
        try:
            out[i] = [float(p) for p in parts]
        except ValueError:
            continue
    return out


def population(mask: np.ndarray, source: str) -> dict:
    indices = np.flatnonzero(mask).astype(int).tolist()
    return {"count": len(indices), "source": source, "indices": indices}


def build_populations(kept: pl.DataFrame, neck: pl.DataFrame) -> dict[str, dict]:
    sub = kept["cell_sub_class"].fill_null("")
    side = kept["side"].fill_null("")
    cell_type = kept["cell_type"].fill_null("")
    ids = kept["id"].to_numpy()
    pops: dict[str, dict] = {}

    for name, classes in (("jo_sound", JO_SOUND), ("jo_wind", JO_WIND)):
        for hand in ("left", "right"):
            mask = sub.is_in(classes).to_numpy() & (side == hand).to_numpy()
            pops[f"{name}_{hand}"] = population(mask, f"cell_sub_class in {classes}, side={hand}")

    descending = (kept["super_class"] == "descending").to_numpy()
    pops["dn_all"] = population(descending, "super_class == descending")

    by_id = dict(zip(neck["id"].to_numpy(), neck["super_cluster"].to_list(), strict=True))
    clusters = np.array([by_id.get(i, "") for i in ids])
    for name, cluster in SUPER_CLUSTERS.items():
        in_cluster = clusters == cluster
        pops[name] = population(in_cluster, f"neck super_cluster == {cluster!r}")
        # The readout steers on left-minus-right, so every cluster is also split.
        for hand in ("left", "right"):
            pops[f"{name}_{hand}"] = population(
                in_cluster & (side == hand).to_numpy(),
                f"neck super_cluster == {cluster!r}, side={hand}",
            )

    for dn in NAMED_DNS:
        for hand in ("left", "right"):
            mask = (cell_type == dn).to_numpy() & (side == hand).to_numpy()
            pops[f"dn_{dn}_{hand}"] = population(mask, f"cell_type == {dn}, side={hand}")

    # Sugar GRNs carry their receptor list in cell_function_detailed
    # ("sugar, Gr5a", "sugar, Gr43a, Gr64f, Gr5a", ...), not in cell_function.
    detailed = kept["cell_function_detailed"].fill_null("")
    is_sugar = detailed.str.starts_with("sugar").to_numpy()
    pops["sugar_grn"] = population(is_sugar, "cell_function_detailed starts with 'sugar'")
    pops["mn_proboscis"] = population(
        (kept["cell_class"].fill_null("") == "proboscis_motor_neuron").to_numpy(),
        "cell_class == proboscis_motor_neuron",
    )
    return pops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=META)
    ap.add_argument("--neck", type=Path, default=NECK)
    ap.add_argument("--out", type=Path, default=Path("data/banc_anatomy.npz"))
    ap.add_argument("--populations", type=Path, default=Path("data/populations.json"))
    args = ap.parse_args()

    df = pl.read_parquet(args.meta)
    log.info("metadata: %d rows, %d columns", len(df), len(df.columns))

    has_class = df["super_class"].is_not_null()
    proofread = df["proofread"].fill_null(False) | df["roughly_proofread"].fill_null(False)
    log.info(
        "filters: %d have a super_class, %d are (roughly) proofread, %d are both",
        has_class.sum(),
        proofread.sum(),
        (has_class & proofread).sum(),
    )
    kept = df.filter(has_class & proofread).with_columns(
        pl.col("root_888").cast(pl.Int64).alias("id")
    )

    soma = parse_positions(kept["nucleus_position_nm"])
    missing = ~np.isfinite(soma).all(axis=1)
    log.info("kept %d neurons; %d without a soma position", len(kept), int(missing.sum()))

    transmitters = kept["neurotransmitter_predicted"].to_list()
    sign = np.array([transmitter_sign(t) for t in transmitters], np.int8)
    log.info(
        "transmitter sign: %d excitatory, %d inhibitory, %d unknown (assumed excitatory)",
        int((sign > 0).sum()),
        int((sign < 0).sum()),
        sum(1 for t in transmitters if not t),
    )

    labels = {
        "super_class": kept["super_class"],
        "cell_class": kept["cell_class"],
        "cell_sub_class": kept["cell_sub_class"],
        "cell_type": kept["cell_type"],
        "cell_function": kept["cell_function"],
        "side": kept["side"],
        "region": kept["region"],
        "neurotransmitter": kept["neurotransmitter_predicted"],
    }
    arrays = {
        name: series.fill_null("").to_numpy().astype("U40") for name, series in labels.items()
    }
    assert set(arrays) == set(CATEGORICAL)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, ids=kept["id"].to_numpy(), sign=sign, soma_xyz=soma, **arrays
    )
    log.info("wrote %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)

    neck = pl.read_csv(args.neck).with_columns(pl.col("id").cast(pl.Int64))
    pops = build_populations(kept, neck)
    with open(args.populations, "w") as f:
        json.dump(pops, f, indent=1)
    log.info("wrote %s", args.populations)
    for name, entry in pops.items():
        log.info("  %-24s %6d  (%s)", name, entry["count"], entry["source"])


if __name__ == "__main__":
    main()
