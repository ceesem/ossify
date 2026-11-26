# Compartment Models

Ossify ships trained classifiers that label each skeleton vertex as axon or dendrite, and the pieces to train your own. Unlike [synapse flow](algorithms_and_analysis.md#compartment-classification), which makes one cut on the whole cell, a model reads local features around each vertex, so it still works when a cell has few synapses or has other cells merged onto it.

## Labelling a cell

```python
from ossify.compartments import describe_models, load_labeler

describe_models()                           # what is bundled, and what each is for

labeler = load_labeler("v1dd_distance")     # a proofread v1dd cell
labels = labeler.predict(cell)              # "axon" / "dendrite" per skeleton vertex

cell.skeleton.add_feature(labels == "axon", name="is_axon")
```

The cell needs `pre_syn` and `post_syn` annotations linked to the skeleton, and the level-2 properties `area_nm2`, `size_nm3` and `max_dt_nm` on its graph. Cells from `load_cell_from_client` and `load_cell_from_l2graph` have both when loaded with `synapses=True`.

To label a cell as it is built:

```python
cell = ossify.load_cell_from_l2graph(
    root_id, client, synapses=True, compartment_labeler=load_labeler("v1dd_local")
)
```

## Choosing a model

Pick by whether the cell is proofread.

| Model | Use it for |
|---|---|
| `minnie65_distance`, `v1dd_distance` | **Proofread cells.** Looks up to 100 µm downstream, which keeps it accurate when synapses are missing. |
| `minnie65_local`, `v1dd_local` | **Objects that may hold merges**, where you want the dendrite of the cell at the soma. Looks no further than 30 µm downstream and uses nothing measured against the whole object, so a merge is seen only close to where it is. |
| `minnie65_ds15_us0_bd0`, `v1dd_ds15_us0_bd0` | The earlier models, kept for comparison. |

The two kinds want opposite things. A long window rescues cells with few synapses but blurs where a merge is. Short windows keep a merge sharp but need most of the cell's synapses to be present: do not use a `local` model on a cell that is missing most of them.

## How well they work

Scored after the decode on proofread cells that neither new model trained on (90 for minnie65, 32 for v1dd), against synapse flow on the full cell. Figures are shares of cable length, averaged over cells.

| Case | `distance` | `local` | Earlier 15-hop |
|---|---|---|---|
| Clean cell: cable labelled correctly | 99.9% | 99.95% or better | 99.95% or better |
| 20% of synapses kept: cable labelled correctly (minnie65 / v1dd) | 99.9% / 99.8% | 98.4% / 97.9% | 96.7% / 94.4% |
| Axon pieces merged on: the cell's own dendrite recovered (minnie65 / v1dd) | 99.6% / 99.3% | 99.9% / 99.9% | 99.6% / 99.4% |

On a clean proofread cell at its measured synapse density all three agree to within a tenth of a percent. The `distance` models earn their place when synapses are missing, and the `local` models when axon is merged onto the cell.

Full tables with the worst single cell, how the models were trained and their known limits are in [`src/ossify/compartment_models/README.md`](https://github.com/ceesem/ossify/blob/main/src/ossify/compartment_models/README.md).

!!! note "What the scores are measured against"
    Synapse flow makes one cut per cell. Where a short axon continues off a dendrite tip, a model can be right and the label wrong, and that counts against the model. The merges are also synthetic: clean axon pieces attached to dendrites. Real unproofread objects have not been scored.

## How a labeler works

A labeler runs three steps: **encode**, **decode**, **clean**.

1. **Encode.** A `ProbaVertexModel` computes per-vertex features and asks the classifier for a probability of each class.
2. **Decode.** A `TransitionSchema` gives a cost for each change of label from a vertex to its child. An exact search over the tree finds the labelling that best fits the probabilities while paying those costs. With the root as dendrite, a cheap dendrite→axon change and an expensive axon→dendrite one, a neuron gets few axons and an axon rarely turns back into dendrite.
3. **Clean.** Decoded compartments of only a few vertices are absorbed into their parent.

`load_labeler` builds all three from the model's config. To use other costs:

```python
from ossify.structured_prediction import TransitionSchema

schema = TransitionSchema(
    classes=["dendrite", "axon"],
    transitions={("axon", "dendrite"): 50, ("dendrite", "axon"): 2},
    root_classes=["dendrite"],
)
labeler = load_labeler("v1dd_distance", schema=schema, absorb_min_size=20)

labels, edge_cost = labeler.predict_with_violations(cell)   # nonzero costs mark paid transitions
```

For the classifier's call alone, with no decode:

```python
from ossify.compartments import ProbaVertexModel

encoder = ProbaVertexModel.from_config("v1dd_distance")
is_axon = encoder.predict(cell) == 1
```

## Features

Features are computed per skeleton vertex and then aggregated over windows of the skeleton around it. The bundled models use windows that reach **downstream only** (toward the tips), measured in path distance, so a vertex is judged by what it leads to.

```python
from ossify.compartments import FEATURE_SETS, make_skel_prop_df

spec, windows = FEATURE_SETS["local"]        # or "distance"
features = make_skel_prop_df(cell, downstream_hops=0, feature_spec=spec, windows=windows)
```

| Feature set | Windows | What it adds to the base features |
|---|---|---|
| `"distance"` | 10, 30 and 100 µm downstream | Radius relative to the cell's typical radius at its inputs and at its outputs |
| `"local"` | 10 and 30 µm downstream | Nothing measured against the whole object |

Both hold surface area, volume, radius and cable length, input and output synapse counts, and those counts per unit area and per unit length. A feature set is a list of feature definitions (`MappedFeature`, `CountFeature`, `DerivedFeature`, `ComputedFeature`) and a list of `Window`s, so you can write your own.

## Training your own model

The `notebooks/` folder of the repository holds the notebooks that trained the bundled models:

- `train_compartment_model.ipynb` trains a `distance` model from proofread cells, with synapse thinning so it holds up on sparse cells.
- `train_unproofread_model.ipynb` trains a `local` model, attaching pieces of other cells' axons to each cell's dendrites to stand in for merge errors.
- `label_compartments.ipynb` is for compartments that have no labels yet. It pairs the features with a widget that shows a cell's vertices in feature space beside the cell in 3D, so you can select and mark them by eye, then trains a classifier from what you marked.

Any estimator with `predict_proba` and `classes_` can be the classifier. Name the feature set so the model can be saved and reloaded:

```python
encoder = ProbaVertexModel(fitted_classifier, feature_columns, feature_set="local")
```

A model is a JSON config beside its model file:

```json
{
  "dataset": "v1dd",
  "description": "What the model is for, shown by describe_models().",
  "feature_set": "local",
  "feature_columns": ["area_um2", "..."],
  "model_file": "xgb/v1dd_local.ubj",
  "decode": {"schema": {"classes": ["dendrite", "axon"], "...": "..."}, "absorb_min_size": 10}
}
```

`load_labeler` and `ProbaVertexModel.from_config` accept the name of a bundled model or the path of a config like this one, with `model_file` read relative to the config.

Before using cells as training data, `ossify.algorithms.synapse_compartment_qc` reports how cleanly a labelling separates a cell's outputs from its inputs.

## See also

- [Compartment models API reference](reference/compartments.md)
- [Algorithms: compartment classification by synapse flow](algorithms_and_analysis.md#compartment-classification)
