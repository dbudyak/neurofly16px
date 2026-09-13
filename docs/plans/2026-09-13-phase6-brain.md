# Phase 6 — Connectome brain (BANC) implementation plan

**Goal:** `--behavior brain` replaces the state machine with a spiking model of
the fly's whole central nervous system: sound drives the Johnston's-organ
neurons, activity propagates through the real wiring, descending neurons steer
the body, and a viewer shows the activity as a heat map while she reacts on the
panel. The FSM stays as the fallback.

**Spec:** `PLAN.md` Phase 6 (6.0–6.5); every parameter and its source is in
`docs/banc.md`.

## What gates what

| step | needs | who |
|---|---|---|
| 6.0a anatomy: neurons, populations, soma map | `htem/BANC-project` files, **no login** | me |
| 6.0b wiring: the connection table | Codex sign-in **or** a Dataverse access request | **owner** |
| 6.1 LIF simulator on the 3090 | 6.0b, `torch` (the `brain` extra, ~2.5 GB) | me |
| 6.2–6.3 stimulus and readout | 6.1 | me |
| 6.4 viewer | 6.0a for the map, 6.1 for the activity | me |
| 6.5 integration into the loop | all of the above | me |

6.0a is built first precisely because it needs nothing from anyone: it produces
the map the heat is later painted onto, and it answers the open questions in
`docs/banc.md` ("Confirm on the host") about JO and DN counts.

## Task 1 — Anatomy from public data (`scripts/build_anatomy.py`)

**Files:** `scripts/build_anatomy.py`, `neurofly16px/brain/__init__.py`,
`neurofly16px/brain/anatomy.py`, `tests/test_anatomy.py`

- Download (pinned commit, into the git-ignored `data/banc/`):
  `data/meta/banc_888_meta_20260521.parquet` (66 MB) and
  `data/banc_annotations/v888/banc_neck_functional_classes*.csv`.
- Keep neurons with a `super_class` and `proofread` or `roughly_proofread`;
  record how many fall out at each filter.
- Write `data/banc_anatomy.npz`: `ids` (int64), categorical `super_class`,
  `cell_class`, `cell_type`, `side`, `region`, `neurotransmitter`, `sign`
  (int8 from the transmitter map in `docs/banc.md`), `soma_xyz` (float32 nm,
  NaN where unknown).
- Write `data/populations.json`: the index sets of `PLAN.md` 6.0 —
  `jo_sound_left/right` (JO-A, JO-B), `jo_wind_left/right` (JO-C, JO-E),
  `dn_all`, the `super_cluster` groups, the named DNs (DNp01 giant fibre,
  DNa02, MDN, DNp09, DNa01, DNb01, DNg13), and the `sugar_grn` /
  `mn_proboscis` sanity pair — each with its provenance and count.
- `brain/anatomy.py` is the loader: `Anatomy.load(path)`, `.populations`,
  `.soma_image(width, height)` (positions binned once into a 2-D histogram, the
  projection chosen after looking at the data).
- Tests run on a small synthetic table: filters, sign map, population selection,
  binning shape and that unknown somata are dropped rather than piled at 0.

## Task 2 — Wiring (`scripts/build_connectome.py`) — blocked on 6.0b

Joins the connection table onto the anatomy, applies `min_synapses` (default 5),
and writes `data/banc_v888.npz` with CSR `indptr/indices/weights` by
presynaptic neuron, `weights = sign × synapse_count`. Records the minimum
`syn_count` actually present, so the threshold the export already applied is
documented rather than assumed.

## Task 3 — LIF simulator (`brain/lif.py`)

Exactly `PLAN.md` 6.1: state `v`, `g`, refractory counters, an 18-slot delay
ring, event-driven propagation over the CSR rows of the neurons that spiked, and
the runaway-excitation guard that halves `weight_scale` when more than 5 % of
neurons spike in a step. Parameters from `docs/banc.md`, all in config. Tests:
silence without input, the sugar-GRN → proboscis-motor-neuron sanity path, seed
determinism, and the population cap firing.

## Task 4 — Stimulus, readout, viewer, integration

`PLAN.md` 6.2–6.5, in that order. The viewer is a `websockets` server plus a
static canvas page showing the soma heat map, per-region bars, the audio
features, the descending-neuron readout and the brain/body time ratio.

## Done when

A clap produces a visible burst propagating through the heat map and a hop on
the Ditoo; sustained noise produces walking; the sugar-GRN sanity test passes;
and the brain/body lag is logged rather than silently accumulating.
