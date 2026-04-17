# Core Concepts

This page introduces the key ideas behind ossify. If you're already familiar with EM reconstruction data (meshes, skeletons, synapse tables), you can skip ahead to [Getting Started](getting_started.md). If not, this page will give you the mental model you need to use ossify effectively.

## Representations of a Neuron

In electron microscopy (EM) connectomics, a single neuron can be represented in several different ways. Each captures different aspects of the cell's structure, and they're useful for different things.

### Surface Mesh

A **mesh** is the full 3D surface of a neuron, reconstructed from the EM segmentation. It consists of millions of vertices connected by triangular faces — essentially the same kind of data you'd see in a 3D modeling program.

Meshes are the most detailed geometric representation. They capture surface area, volume, and fine shape features. However, because neurons are long and branching, meshes are unwieldy for reasoning about topology (which branch is connected to what).

### Skeleton

A **skeleton** is a simplified tree structure that runs through the center of the neuron. It has thousands to low tens of thousands of vertices connected by edges, with a single designated root (typically at the soma). Every vertex has exactly one parent, except the root.

Skeletons are the most natural representation for morphological analysis. They make it easy to measure cable length, identify branch points, compute distances along the arbor, and classify branches into compartments (axon, dendrite, etc.).

### Graph

A **graph** is the fundamental representation of neuronal topology in EM analysis databases like CAVE. It has roughly twice as many vertices as the skeleton and represents the spatial connectivity of the neuron without imposing a tree structure — graphs can have cycles or more complex connectivity.

Graphs are first-class in ossify because they make it easy to bring in the raw database representation directly. They're also a generic way to embed objects in space with continuous topology, without the structural assumptions that skeletons require.

### Point Annotations

**Annotations** are sparse points with metadata — most commonly synapses, but also branch labels, manual markers, or any other per-location data. A typical neuron might have thousands to tens of thousands of synapses, each with features like type, size, and partner identity.

Annotations don't define the shape of the neuron. They decorate it with biologically critical information.

### You Don't Need All of Them

Ossify is flexible about which representations you use. A `Cell` with just a skeleton and synapse annotations works fine. So does one with a skeleton, graph, mesh, and multiple annotation types. Use what you have and what your analysis needs.

## The Linking Problem

These representations describe the *same physical object*, but they have different vertex sets. A skeleton vertex is not the same thing as a mesh vertex, even if they're near the same location in space.

This creates a practical problem: many analyses require combining information across representations. For example:

- "What is the total synapse count on each skeleton branch?"
- "What compartment label does each mesh vertex belong to?"
- "What's the total volume (from the mesh) associated with each skeleton vertex?"

### Why Not Just Nearest-Neighbor?

The naive solution — map each point to the nearest vertex in the other representation — is usually right. But neurons have tortuous, winding structure. A synapse on one branch might be spatially close to a nearby but *topologically distant* branch. Nearest-neighbor mapping will silently assign it to the wrong branch, and this happens often enough to be a real problem in quantitative analysis.

### Links: Ossify's Solution

Ossify uses explicit **Links** — precomputed mappings between vertex sets that fully preserve the correct correspondence. When you load a cell from CAVE or build one with known relationships, these mappings are established once and then used automatically whenever you move data between layers.

With a link in place, you can:

- **Map features** between layers (e.g., aggregate mesh surface area onto skeleton vertices)
- **Map indices** to find corresponding vertices across representations
- **Propagate masks** so that filtering one layer automatically filters linked layers

This is the core differentiator of ossify: it makes cross-representation analysis reliable and easy.

## The Cell Container

A **Cell** is ossify's central object. It holds:

- **Layers** — mesh, skeleton, and/or graph representations of the neuron
- **Annotations** — point clouds like synapses, each with their own features
- **Links** — mappings connecting layers and annotations to each other

All of these share a coordinate space. When you mask a layer (e.g., filter the skeleton to dendrite-only), linked annotations and features update accordingly.

```python
import ossify as osy

cell = osy.load_cell('https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy')
cell.describe()  # See everything the cell contains
```

## Compartments and the SWC Standard

Neurons are divided into functional compartments: soma, axon, dendrite, and sometimes finer subdivisions. In ossify, compartment labels follow the [SWC format](http://www.neuronland.org/NLMorphologyConverter/MorphologyConverter/MorphologyFormats/SWC.html) standard, stored as integer features on skeleton vertices:

| Value | Compartment |
|-------|-------------|
| 0     | Undefined   |
| 1     | Soma        |
| 2     | Axon        |
| 3     | Dendrite    |

These labels are used throughout ossify for filtering, visualization, and analysis. For example, you can mask a cell to dendrite-only with a single line:

```python
dendrite_mask = cell.skeleton.features['compartment'] == 3
```

## Where Does the Data Come From?

Ossify works with data from several sources:

- **CAVE / CAVEclient** — the database system hosting large EM datasets like [MICrONS](https://www.microns-explorer.org/) and FlyWire. Ossify can load cells directly from CAVE with `load_cell_from_client()`.
- **`.osy` files** — ossify's native format, which preserves all layers, annotations, and links. Files can be local or hosted remotely (including cloud storage).
- **Raw arrays** — you can build a Cell from scratch using numpy arrays and pandas DataFrames, useful for custom pipelines or non-CAVE datasets.
- **Legacy MeshWork files** — ossify can import `.h5` files from MeshParty's MeshWork format.

You don't need CAVE access to use ossify. The [Getting Started](getting_started.md) tutorial uses a hosted `.osy` file that anyone can load.

## Next Steps

- [Getting Started](getting_started.md) — hands-on tutorial with the example cell
- [Linking and Mapping](linking_and_mapping.md) — deep dive into ossify's cross-representation mapping
- [The Cell Object](cell_object.md) — full reference for creating and managing cells
