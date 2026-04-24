**ossify**
- **What**: Neuronal morphology analysis library — loading, manipulating, visualizing, and analyzing neuron meshes, skeletons, and synaptic annotations. Built around a `Cell` container with multiple `Layer` types (surface meshes, skeleton trees, point cloud annotations) that share a common coordinate framework.
- **Where**: `pip install ossify` / [PyPI](https://pypi.org/project/ossify/) / [Docs](https://www.csdashm.com/ossify/) / [GitHub](https://github.com/ceesem/ossify)
- **Key capabilities**: Translating features and metadata between representations (e.g. mesh ↔ skeleton ↔ synapse annotations), so you don't have to rewrite the mapping logic yourself. Also provides 2D skeleton visualization for quick inspection and figures.
- **Data sources**: Designed for CAVE-hosted datasets (MICrONS, FlyWire) via `caveclient`, but the core data structures are source-agnostic.
- **When to use**: You need to work with neuron morphology at synaptic resolution and want to move between mesh, skeleton, and annotation views of the same cell without rolling your own conversion code. Roughly feature-complete with MeshParty.
- **When not to use**: Pure connectomics queries with no morphology component (use `caveclient` directly), or if you only need simple skeleton I/O (use `cloud-files` + `trimesh`.
- **Status**: v0.0.4, early but usable. API may shift. Docs are a work in progress.
- **Owner**: Casey Schneider-Mizell
