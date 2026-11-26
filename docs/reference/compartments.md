# Compartment Models

Trained per-vertex classifiers, the tree decode that turns their output into compartments, and the feature extraction behind them. See the [user guide](../compartment_models.md) for how they fit together.

## Overview

| Category | Members | Purpose |
|----------|---------|---------|
| **[Loading](#loading)** | `load_labeler`, `describe_models`, `get_models`, `load_model_config` | Bundled models and their configs |
| **[Labelers](#labelers)** | `StructuredLabeler`, `ProbaVertexModel` | Encode, decode and clean |
| **[Decoding](#decoding)** | `TransitionSchema`, `decode_tree`, `absorb_small_compartments` | Label priors and exact tree decode |
| **[Features](#features)** | `make_skel_prop_df`, `Window`, `WindowContrast`, `FEATURE_SETS` | Per-vertex features and windows |
| **[Quality control](#quality-control)** | `synapse_compartment_qc` | How cleanly a labelling separates outputs from inputs |

---

## Loading {: .doc-heading}

::: ossify.compartments.load_labeler
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

::: ossify.compartments.describe_models
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

::: ossify.compartments.get_models
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

::: ossify.compartments.load_model_config
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

---

## Labelers {: .doc-heading}

::: ossify.compartments.StructuredLabeler
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

::: ossify.compartments.ProbaVertexModel
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

---

## Decoding {: .doc-heading}

::: ossify.structured_prediction.TransitionSchema
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

::: ossify.structured_prediction.decode_tree
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

::: ossify.structured_prediction.absorb_small_compartments
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

---

## Features {: .doc-heading}

::: ossify.compartments.make_skel_prop_df
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

::: ossify.compartments.Window
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

::: ossify.compartments.WindowContrast
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

### Feature sets

`ossify.compartments.FEATURE_SETS` maps a name to a feature spec and its windows. A model config names one with its `feature_set` key.

| Name | Feature spec | Windows |
|---|---|---|
| `"distance"` | `DISTANCE_FEATURE_SPEC` | `DISTANCE_WINDOWS`: 10, 30 and 100 µm downstream |
| `"local"` | `LOCAL_FEATURE_SPEC` | `LOCAL_WINDOWS`: 10 and 30 µm downstream |

A config with no `feature_set` uses `DEFAULT_FEATURE_SPEC` with its hop settings.

::: ossify.compartments.cable_length_feature
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

::: ossify.compartments.reference_radius_feature
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false

---

## Quality control {: .doc-heading}

::: ossify.algorithms.synapse_compartment_qc
    options:
        heading_level: 3
        show_root_heading: true
        show_root_full_path: false
