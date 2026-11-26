# Compartment models

Bundled axon/dendrite classifiers for skeletons built from CAVE level-2 graphs. Each
model is a JSON config in this folder plus an XGBoost file in `xgb/`.

## Use

```python
from ossify.compartments import describe_models, load_labeler

describe_models()                          # what is bundled, and what each is for

labeler = load_labeler("v1dd_distance")    # a proofread v1dd cell
labels = labeler.predict(cell)             # "axon" / "dendrite" per skeleton vertex
```

`load_labeler` returns the classifier together with the tree decode stored in the
config's `decode` block. The cell needs `pre_syn` and `post_syn` annotations and the
level-2 properties (`area_nm2`, `size_nm3`, `max_dt_nm`) on its graph. For the
classifier alone, without the decode, use `ProbaVertexModel.from_config(name)`.

## Which model

| Model | Use it for | Features |
|---|---|---|
| `minnie65_distance`, `v1dd_distance` | Proofread cells | Downstream windows of 10, 30 and 100 µm, counts per area and per length, radius relative to the cell's own (52 features) |
| `minnie65_local`, `v1dd_local` | Objects that may hold merges, where the goal is the dendrite of the cell at the soma | Downstream windows of 10 and 30 µm only, nothing measured against the whole object (33 features) |
| `minnie65_ds15_us0_bd0`, `v1dd_ds15_us0_bd0` | Comparison with earlier results | Sums over 15 hops downstream (12 features) |

The two new kinds want opposite things. A long window rescues cells with few synapses
but blurs where a merge is; short windows keep a merge sharp but need the cell's
synapses to be mostly present. Pick by whether the cell is proofread.

All windows are downstream only, so a vertex is judged by what it leads to. That is
what lets an axon leaving a dendrite, real or merged, be picked up where it starts.

## How they were trained

Trained on 2026-11-26 by the notebooks in `notebooks/` (`train_compartment_model*.ipynb`
for `distance`, `train_unproofread_model*.ipynb` for `local`).

| | minnie65 | v1dd |
|---|---|---|
| Datastack, version | `minnie65_public` v1822 | `v1dd` v1643 |
| Proofread cells (axon and dendrite) with a skeleton | 2,219 | 1,169 |
| Passed label QC | 1,969 | 1,018 |
| Training / held-out cells | 1,575 / 394 | 814 / 204 |

- **Labels** are the skeleton service's `compartment` feature, made by synapse flow on
  the proofread cell. Nothing was labelled by hand.
- **Label QC**: a cell is used only if at most 10% of its outputs sit on dendrite and at
  most 10% of its inputs on axon, it has at least 100 of each, its root is a soma, and
  2% to 98% of its vertices are axon.
- **`distance` models**: every cell is featurized as it is and once more with a random
  share of its synapses removed (50%, 20%, 10% or 5% kept), labels unchanged. Rows are
  weighted by the inverse square root of how common their cell type is, and the type
  with the sparsest dendritic inputs (5P-NP / L5NP) is featurized at every thinning
  level. Cell type only shapes the training data; it is never a feature.
- **`local` models**: every cell is used as it is and once more with ten pieces of
  other cells' axons (20 to 300 µm of cable each) attached to random dendrite vertices,
  standing in for merge errors. A cell only receives pieces from its own side of the
  train / held-out split. No synapse thinning.
- **Decode** (in each config's `decode` block): root is dendrite, dendrite→axon costs 2,
  axon→dendrite costs 50, and decoded compartments of 10 vertices or fewer are absorbed
  into their parent.

## Expected quality

Scored on cells neither new model trained on (90 for minnie65, 32 for v1dd), after the
decode, against synapse flow on the full proofread cell. Shares are of cable length;
the figure in brackets is the worst single cell.

**Cable labelled correctly**

| Case | Dataset | `distance` | `local` | 15-hop |
|---|---|---|---|---|
| Clean proofread cell | minnie65 | 99.91% (98.45%) | 99.97% (99.20%) | 99.95% (99.12%) |
| | v1dd | 99.92% (98.48%) | 99.95% (99.71%) | 99.97% (99.70%) |
| 20% of synapses kept | minnie65 | 99.92% (98.03%) | 98.39% (93.25%) | 96.72% (85.57%) |
| | v1dd | 99.77% (93.73%) | 97.92% (86.66%) | 94.38% (75.24%) |

**The cell's own dendrite recovered**

| Case | Dataset | `distance` | `local` | 15-hop |
|---|---|---|---|---|
| Ten axon pieces attached | minnie65 | 99.58% (95.25%) | 99.90% (95.30%) | 99.64% (95.25%) |
| | v1dd | 99.25% (93.91%) | 99.92% (98.20%) | 99.43% (95.69%) |
| 20% of synapses kept | minnie65 | 99.76% (92.31%) | 95.21% (71.61%) | 90.57% (59.61%) |
| | v1dd | 99.84% (97.42%) | 94.25% (61.98%) | 84.46% (29.43%) |

What to take from this:

- On a clean proofread cell at its measured synapse density, all three are within a
  tenth of a percent of each other.
- The `distance` models hold up when synapses are missing; the others do not.
- The `local` models are the best at recovering the dendrite when axon is merged on,
  and they lift the worst cell most in v1dd.
- The `local` models should not be used on cells missing most of their synapses. With
  20% kept, the worst cells lose a third or more of their dendrite.

## Limits

- **Scores are against synapse flow, which makes one cut per cell.** Where a short axon
  continues off a dendrite tip, the models can be right and the label wrong, and that
  counts against them here.
- **The merges are synthetic.** They are clean axon pieces of at most 300 µm on
  dendrites. Merged dendrites from other cells, long merged axons and real unproofread
  objects have not been tested.
- **Only cells with a soma at the root.** The decode assumes the root is dendrite.
- **Cells with sparse dendritic inputs are the hard case** (5P-NP in minnie65, L5NP in
  v1dd). They are few in the training data: 12 and 24 proofread cells.
- **Excitatory and inhibitory cells are pooled.** No cell type is used at prediction.

## Adding a model

A config needs `model_file` (relative to the config), `feature_columns`, and either the
hop settings or a `feature_set` naming an entry in `ossify.compartments.FEATURE_SETS`.
Add `dataset` and `description` so it shows in `describe_models()`, and a `decode` block
so `load_labeler` can build the labeler. The training notebooks write all of these. A
feature set that changes should get a new name, since a saved model is tied to the
features it was trained on.
