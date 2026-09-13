# BANC connectome and the LIF model — verified facts

Verified 2026-09-12. `file:line` references point into clones of
`htem/BANC-project` (commit `e31a2e2`, 2026-07-20) and
`philshiu/Drosophila_brain_model` (commit `91bdd1e`, 2024-09-14).

## Dataset

- BANC v888: ~188,000 neurons, ~199 million predicted synapses, brain +
  suboesophageal zone + cervical connective + entire VNC, one adult female
  (`BANC-project/README.md`, "About the BANC"). Paper: Bates, Phelps, Kim,
  Yang et al., *Nature* 2026, doi:10.1038/s41586-026-10735-w (open access).
- FlyWire Codex lists v888 as 158,262 neurons and 3,037,361 connections
  (`codex.flywire.ai/api/download?dataset=banc`). Codex downloads require a
  Google sign-in. The connection count is consistent with a ≥ 5-synapse
  threshold on pre→post pairs (the unfiltered v2 edgelist has 11.5 M pairs);
  confirm the threshold from the downloaded file's minimum `syn_count`.
- Harvard Dataverse doi:10.7910/DVN/7WTH1N (version 3.0, CC BY 4.0).
  ~~Anonymous downloads return HTTP 403; the dataset has `fileAccessRequest:
  true`, so a Dataverse account and an access request are needed.~~
  **Wrong — corrected 2026-09-13:** the files we need report
  `restricted: false` and download anonymously; see "Wiring" below for the
  one-line `curl`. Files we care about (API listing, 2026-09-12):

  | path | rows × cols | size | file id |
  |---|---|---|---|
  | `compiled_data/banc_888_meta.feather` | 188,162 × 79 | 57.5 MB | 14033740 |
  | `compiled_data/banc_888_edgelist_simple_v2.feather` | 11,510,975 × 6 (v2 synapses, size ≥ 5 voxels) | 305 MB | 13992792 |
  | `compiled_data/banc_888_edgelist_simple_v3.feather` | 13,507,098 × 6 (v3 synapses, size ≥ 10 voxels) | 359 MB | 13918810 |
  | `compiled_data/banc_888_neurotransmitter_prediction_v2.csv` | one row per presynaptic neuron, 17 cols | 21 MB | 13916450 |

  Per-synapse tables (12–20 GB) and influence shards (restricted) are not
  needed.
- Available **without any login** from the `htem/BANC-project` git repo:
  - `data/meta/banc_888_meta_20260521.parquet` — 66 MB, one row per neuron,
    ~165 columns, the full metadata snapshot used by the paper's scripts
    (`data/meta/README.md`).
  - `data/banc_annotations/v888/banc_neck_functional_classes.csv` — 3,161
    AN/DN rows; columns include `id`, `neurotransmitter`, `side`, `region`,
    `super_class`, `hemilineage`, `cell_function`, `cell_class`, `cell_type`,
    `super_cluster`, `cluster`.
  - `data/banc_annotations/v888/banc_neck_functional_classes_by_neuron.csv`
    — 3,693 rows: `id`, `super_class`, `cell_type`, `side`, `cluster`.
  - `data/banc_annotations/annotation_terms_list.xlsx` — the controlled
    vocabulary for every taxonomy column.

  The neuron ids in these files are v888 root ids, so they join directly
  with the Codex / Dataverse tables.

## Annotation vocabulary (from `annotation_terms_list.xlsx`)

- `flow`: afferent, efferent, intrinsic.
- `super_class`: ascending, descending, sensory, motor, central_brain_intrinsic,
  ventral_nerve_cord_intrinsic, optic_lobe_intrinsic, visual_projection,
  visual_centrifugal, sensory_ascending, sensory_descending,
  visceral_circulatory, glia, ascending_visceral_circulatory.
- `cell_class` (relevant subset): `johnstons_organ_A_neuron`,
  `johnstons_organ_B_neuron`, `johnstons_organ_C_neuron`,
  `johnstons_organ_D_neuron`, `johnstons_organ_E_neuron`,
  `johnstons_organ_F_neuron`, `johnstons_organ_other_neuron`,
  `descending_neuron`, `ascending_neuron`, `leg_motor_neuron`,
  `wing_motor_neuron`, `neck_motor_neuron`, `taste_bristle_gustatory_neuron`,
  `proboscis_motor_neuron`.
- `cell_function` includes `auditory`, `jump_escape`, `walking`, `steering`,
  `flight`, `grooming`, `landing`, `halting`, `escape_takeoff`,
  `threat_response`; `cell_function_detailed` includes
  `auditory_high_frequency`, `auditory_low_frequency`.
- `neurotransmitter_predicted`: acetylcholine, dopamine, gaba, glutamate,
  histamine, octopamine, serotonin, tyramine.
- `side`: left, right, midline. `region`: brain, central_brain,
  neck_connective, optic_lobe, ventral_nerve_cord.

## Descending neurons in v888 (from `banc_neck_functional_classes.csv`)

1,313 DNs (1,314 in the by-neuron file). `super_cluster` sizes: head
orienting 250, probing 189, flight power 181, walking 159, grooming 101,
takeoff-landing 97, flight steering 1 88, visceral control 49, threat
response 48, flight steering 2 45, reproduction 33, feeding 32, taste-touch
20, tactile 15, vibratory 6. `cell_function` on DNs: flight 66, steering 40,
grooming 35, reaching 23, walking 20, escape_takeoff 14, landing 10,
halting 2. Transmitters: 924 acetylcholine, 234 gaba, 98 glutamate.

Named DNs present (one per side unless noted): `DNp01` (giant fibre, cluster
DN_05), `DNa02` (DN_14), `MDN` (4 cells, DN_14), `DNp09` (DN_02), `DNa01`
(DN_02), `DNb01` (DN_12), `DNg13` (DN_02), `DNp06` (DN_08), `DNa11`, `DNp07`,
`DNp10`, `DNp11`, `DNa03`, `DNa04`, `DNa05`, `DNa08`, `DNb05`, `DNb06`,
`DNg01`…`DNg05` (multi-cell), `DNpe001`…`DNpe012`. 472 distinct DN cell types.

## Johnston's organ neurons

Cell classes `johnstons_organ_{A,B,C,D,E,F,other}_neuron`, split by `side`.
JO-A and JO-B respond to antennal vibration (sound); JO-C/JO-E to static
deflection (wind, gravity); JO-D/JO-F other. Counts per class come from the
metadata parquet on the host (not read here: this Mac has no pyarrow and was
out of disk). Shiu et al. stimulated JO-CE and JO-F populations in the FAFB
model and read out `aBN1` (antennal grooming)
(`Drosophila_brain_model/figures.ipynb` cells 50–72), which shows JO drive
propagates in this model class.

## Shiu et al. 2024 LIF model

Paper: *A Drosophila computational brain model reveals sensorimotor
processing*, Nature 2024, doi:10.1038/s41586-024-07763-9 (PMC11446845).

Parameters (`Drosophila_brain_model/model.py:15-53`):

| symbol | value | meaning |
|---|---|---|
| `v_0` | −52 mV | resting potential |
| `v_rst` | −52 mV | reset after spike |
| `v_th` | −45 mV | threshold |
| `t_mbr` | 20 ms | membrane time constant (2 µF/cm² × 10 kΩ·cm²) |
| `tau` | 5 ms | synaptic (alpha) time constant |
| `t_rfc` | 2.2 ms | refractory period |
| `t_dly` | 1.8 ms | synaptic delay |
| `w_syn` | 0.275 mV | weight per synapse; the only free parameter, tuned so 100 Hz sugar GRN drive gives ~80 % of maximal MN9 firing |
| `r_poi` | 150 Hz | default stimulation rate |
| `f_poi` | 250 | Poisson weight multiplier |

Equations (`model.py:44-52`):

```
dv/dt = (v_0 - v + g) / t_mbr        (frozen during refractory)
dg/dt = -g / tau                     (frozen during refractory)
spike when v > v_th;  reset: v = v_rst, g = 0
on presynaptic spike, after t_dly:  g += w_ji
w_ji = sign(j) * synapse_count(j -> i) * w_syn      (model.py:175-183)
```

Brian2 defaults apply: `dt = 0.1 ms`, exact ("linear") integration.

Sign convention (paper Methods, "Neurotransmitter predictions"): a
presynaptic neuron is inhibitory (−1) if more than half of its presynaptic
sites are predicted GABA or glutamate; acetylcholine, dopamine, octopamine and
serotonin are excitatory (+1). Cleft-score cutoff 50. Each neuron is
exclusively excitatory or inhibitory. Glutamate as inhibitory is an explicit
assumption they tested by flipping it. BANC additionally predicts histamine
and tyramine: treat histamine as inhibitory (photoreceptor transmitter acting
on HisCl channels) and tyramine as excitatory; both are config entries.

Synapse threshold: the paper states none ("All 127,400 proofread neurons from
Flywire materialization v.630 are included"). BANC's `influencer` package
defaults to `count_thresh = 5`. We keep `min_synapses` in config, default 5,
and test 1 during tuning.

Stimulation (`model.py:58-106`): a `PoissonInput` with `N=1`, rate `r_poi`
and weight `w_syn * f_poi = 68.75 mV` (far above the 7 mV threshold gap), with
the target's refractory period set to 0, so the target fires at ≈ `r_poi`.
Our equivalent: emit Bernoulli(`rate * dt`) spikes directly in the stimulated
neurons.

Basal firing rate is 0, so inhibitory input only matters where there is
drive; absolute rates are not meaningful, differences between conditions are.

## Sizing for the GPU implementation

- N ≈ 158k (Codex) to 188k (all segments). Filter to neurons with a
  `super_class` and `proofread` or `roughly_proofread` set; record the count.
- Edges: 3.0 M (≥ 5 synapses) to 11.5 M (all pairs). CSR float32 + int32 →
  24–92 MB. Trivial for a 24 GB card.
- A dense SpMV per 0.1 ms step over 11.5 M nonzeros would need ~1e11
  nonzero-ops/s for real time; a 3090 delivers ~1–2e10 for SpMV. Real time
  therefore needs event-driven propagation: gather only the outgoing rows of
  the neurons that spiked this step (tens to hundreds per step at plausible
  rates). Kernel-launch overhead (~10 launches × ~10 µs) then bounds the rate
  at ≈ 10k steps/s, i.e. real time is borderline; the plan decouples brain
  time from body time and logs the ratio.
- Delay ring buffer: `t_dly / dt = 18` slots; refractory = 22 steps.

## Confirmed on the host (Phase 6.0) — all three answered below

1. ✅ Export file names, columns and threshold — "Wiring", below. No access
   approval was needed, and the export is *not* pre-thresholded: 11,752,828
   pairs with `count` from 1.
2. ✅ JO-A..F counts and the DN clusters — "Host results", below. The labels
   are in `cell_sub_class`, not `cell_class`.
3. ✅ Soma positions come from `nucleus_position_nm` (present for 153,468 of
   188,508 rows), parsed from `"x, y, z"` strings; the projection is x against
   y, chosen by measurement rather than by eye — "Host results", below.

## Host results (2026-09-13, from the public tables)

`scripts/fetch_banc.sh` + `scripts/build_anatomy.py` on `banc_888_meta_20260521.parquet`
(188,508 rows x 165 columns) and `banc_neck_functional_classes.csv` (3,161 rows).

**Correction to the notes above:** the Johnston's-organ labels are in
**`cell_sub_class`**, not `cell_class` — for these neurons `cell_class` is
`chordotonal_organ_neuron`. Likewise the sugar GRNs carry their receptors in
`cell_function_detailed` (`"sugar, Gr5a"`, `"sugar, Gr43a, Gr64f, Gr5a"`, ...),
so `cell_function == "sugar"` matches nothing.

Filters: 159,876 neurons have a `super_class`, 155,937 are proofread or roughly
proofread, **144,047 are both** and are kept. 27,497 of those have no soma
position and are dropped from the map (never binned at the origin).

Transmitter signs: 95,789 excitatory, 48,258 inhibitory, 3,464 unknown and
assumed excitatory.

| population | count | selector |
|---|---|---|
| JO sound (A + B) | 200 left, 250 right | `cell_sub_class`, by `side` |
| JO wind (C + E) | 230 left, 224 right | `cell_sub_class`, by `side` |
| descending neurons | 1,316 | `super_class == descending` |
| DN walking / takeoff-landing / threat / head-orienting | 218 / 253 / 135 / 511 | neck `super_cluster` |
| DN flight steering 1 / 2 | 91 / 211 | neck `super_cluster` |
| named DNs (DNp01, DNa02, MDN, DNp09, DNa01, DNb01, DNg13) | 1-2 per side | `cell_type` |
| sugar GRNs | 537 | `cell_function_detailed` starts with `sugar` |
| proboscis motor neurons | 35 | `cell_class` |

Per-class JO counts (both sides): JO-A 93, JO-B 417, JO-C 53, JO-D 13, JO-E 409,
JO-F 187, other 21.

**Projection for the heat map.** y is the body axis: mean soma y is 149 um
(central brain), 157 um (optic lobe) and 778 um (ventral nerve cord), against a
19 um difference in x. So the map projects x against y, unflipped, which draws
the fly head-up with the nerve cord below. Soma extent: x 83-921 um, y 39-1034
um, z 1-315 um.

## Wiring (2026-09-13) — the Dataverse files are NOT access-gated

The note above ("anonymous downloads return HTTP 403 ... an access request is
needed") is wrong for the files we need. `banc_888_edgelist_simple_v2.feather`
(305 MB, id 13992792) has `restricted: false` and

    curl -L -o data/banc/banc_888_edgelist_simple_v2.feather \
        https://dataverse.harvard.edu/api/access/datafile/13992792

downloads it anonymously via a signed S3 redirect. No Dataverse account, no
Codex sign-in. The dataset page is
`https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/7WTH1N`
(the DOI itself resolves to the citation page, not the files).

Schema: `pre`, `post` (root ids as strings), `count` (synapses), `norm`,
`post_count`, `pre_count`. **11,752,828 pairs, `count` from 1 to 1,103** — so the
export is *not* pre-thresholded at 5 as assumed; `min_synapses` is ours to choose.

`scripts/build_connectome.py` with the default `min_synapses = 5`:

| step | pairs |
|---|---|
| all pairs | 11,752,828 |
| `count >= 5` | 1,614,479 (13.7 %) |
| both endpoints among the 144,047 kept neurons | 1,514,402 (93.8 %) |
| autapses dropped | −73,567 |
| **CSR edges** | **1,440,835**, 40.3 % inhibitory, mean \|weight\| 12.3, densest row 2,814 |

**Autapses are dropped by default** (`--keep-autapses` to keep them). They are
larger than real edges (mean 16.2 synapses against 12.3), and for ~11k neurons
the self-edge carries more than half the outgoing weight — segmentation
artefacts. At `w_syn` 0.275 mV a 16-synapse autapse is 4.4 mV against the 7 mV
threshold gap, i.e. a neuron would re-excite itself on its own spike.

Output `data/banc_v888.npz`: `indptr` (n+1), `indices`, `weights`
(sign x synapse count), `ids`, `min_synapses`, `autapses`. 5.3 MB compressed.

## LIF model on the host (2026-09-13, RTX 3090)

`neurofly16px/brain/lif.py`, 144,047 neurons and 1,440,835 edges, dt 0.1 ms.

| propagation | steps/s | real time |
|---|---|---|
| event-driven gather of the spiking rows | 945 | 0.09x |
| same, with the runaway guard removed | 1,224 | 0.12x |
| **sparse matrix-vector product** | **2,949** | **0.29x** |

The plan assumed event-driven gathering would beat an SpMV. It does not, and the
reason is not arithmetic: `nonzero()` and reading the gather size force a
GPU-to-CPU synchronisation on *every* step, while `Wt @ spikes` over 1.4 M edges
is ~11 MB of traffic and needs none. The wiring is therefore stored twice, CSR by
presynaptic neuron (for inspection) and CSR by postsynaptic neuron (for the
SpMV). For the same reason the runaway guard reads the spike count only every
`guard_every_steps` (50), and the Poisson drive is kept as a dense probability
vector rather than an index list -- selecting the stimulated neurons gives the
step a data-dependent shape, which is another sync, and measured *slower* than
drawing one random number per neuron (2,063 vs 2,897 steps/s).

At 339 us/step the loop is launch-bound, not bandwidth-bound: ~16 elementwise
kernels over 144k floats. `torch.compile(mode="reduce-overhead")` halves the
elementwise part (219 -> 110 us) but was not adopted here; 0.29x real time is
within what PLAN.md 6.5 provides for (the body keeps real time, the brain's lag
is logged).

### Sanity check (`scripts/bench_brain.py`)

Sugar GRNs stimulated at 100 Hz for one simulated second:

| population | spikes | per neuron | unstimulated control |
|---|---|---|---|
| proboscis motor neurons | 80 | 2.3 Hz | 0 |
| descending, walking cluster | 126 | 0.6 Hz | 0 |
| all descending neurons | 546 | 0.4 Hz | 0 |
| Johnston's organ, sound, left | 0 | 0.0 Hz | 0 |

Taste drives the proboscis motor neurons it should, does not touch the auditory
neurons it should not, and the resting network is silent. No runaway: the
FAFB-tuned `w_syn` of 0.275 mV holds on brain + nerve cord at `min_synapses` 5.

## What sound actually reaches the body (2026-09-13)

Stimulating the Johnston's-organ sound populations (JO-A + JO-B, both sides) for
one simulated second and counting descending spikes:

| drive | DNp01 (giant fibre) | threat response | takeoff / landing | flight steering | walking |
|---|---|---|---|---|---|
| quiet | 0 | 0 | 0 | 0 | 0 |
| 40 Hz (speech) | 50 / 10 Hz | 0.48 | 0.15 | 0.12 | **0.00** |
| 150 Hz (loud) | 139 / 95 Hz | 1.75 | 1.57 | 1.73 | **0.00** |
| 250 Hz (clap) | 184 / 143 Hz | 4.20 | 3.12 | 2.6 | **0.00** |

The result is unambiguous and biologically sensible: sound reaches the body
through the **escape pathway**. The giant fibre dominates by two orders of
magnitude, then the threat-response and takeoff clusters; the walking cluster is
silent at every level.

This killed the readout as first written (PLAN.md 6.3 mapped `dn_walking` to
forward speed, which would never have moved). The readout now uses what responds:

- **escape**: DNp01 above `giant_fibre_hz`, *and* above `gf_novelty_ratio` times
  its own slow baseline. Without that second condition any sustained sound pins
  the fly in escape forever — the giant fibre keeps firing while the ears are
  driven. Real escape responses habituate, so this is a feature, not a hack.
- **forward**: the mean rate over all 1,316 descending neurons, against
  `drive_hz` (2 Hz = full speed; measured range is 0.2 Hz for speech, ~1 Hz for a
  clap).
- **turn**: the left-right *contrast* of descending activity,
  `(L - R) / (L + R)`, so it does not grow with loudness.

Hold times (escape duration, idle threshold) are wall-clock, not simulated: the
brain runs at a fraction of real time and a 0.8 s escape measured in brain time
would keep the fly airborne for several seconds on the desk.

## Viewer

`neurofly16px/viewer/` serves the heat map over websockets from inside the brain
process (one port, page and stream). Each pixel is a bin of soma positions in the
x-y projection, coloured by spikes per 20 ms window; the panel also shows the
descending rates, the resulting command, and the brain/body speed ratio.

Measured end to end (`--audio demo --behavior brain --sim flybody --device
ditoo --viewer`): silence leaves the map dark, talking scatters activity through
the head, and a clap lights both antennal regions and fires the giant fibre,
which the readout turns into an escape and the body into a hop on the panel. The
brain runs at 0.2x real time while sharing the host with MuJoCo (0.27x alone).
