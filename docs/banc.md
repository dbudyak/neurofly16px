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
  Anonymous downloads return HTTP 403; the dataset has `fileAccessRequest:
  true`, so a Dataverse account and an access request are needed. Files we
  care about (API listing, 2026-09-12):

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

## Confirm on the host (Phase 6.0)

1. Codex export file names and columns for v888, or Dataverse access
   approval; minimum `syn_count` in the connection table.
2. JO-A..F counts per side from the metadata parquet; DN counts by
   `super_cluster` after joining with the connection table (some DNs may
   lack edges).
3. Soma-position column and units (`position` is voxel xyz as a string in the
   CSVs; `root_position_nm` exists too); choose the 2-D projection for the
   heat map after plotting once.

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
