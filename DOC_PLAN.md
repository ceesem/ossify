# Ossify Documentation Plan

## Problem with Current Docs

The existing documentation is reference-oriented and AI-generated. It explains *what* each API does but not *why* someone would use it or how the pieces fit together conceptually. A newcomer who doesn't already understand connectomics reconstruction data (meshes from segmentation, skeletons from mesh simplification, synapse tables from annotation) will struggle to follow along.

## Target Audience

The primary audience is **computational neuroscientists and data scientists** who:

- Work with neuron reconstructions from EM datasets (MICrONS, FlyWire, etc.)
- May be experienced Python users but haven't used ossify or MeshParty before
- Need to do things like: visualize neurons, map features across representations, classify branches, extract morphological/synaptic features at scale for cell typing, filter synaptic connectivity by anatomical properties

The docs should be accessible to someone with strong Python skills who may not know what a "skeleton" or "mesh" means in the connectomics context, while not being patronizing to experienced connectomics researchers.

## Guiding Principles

1. **Concept before API** — Explain the neuroscience data model before showing code.
2. **One real cell throughout** — Use the hosted example cell (`864691135336055529.osy`) as the running example so readers can follow along immediately without CAVE access.
3. **Progressive disclosure** — Start with loading and inspecting, then mapping, then analysis, then custom construction.
4. **Figures everywhere** — Each conceptual section should have at least one plot showing the data being discussed.

---

## Proposed Documentation Structure

### Section 1: Home / Overview (`index.md`) — rewrite

**Goal:** Explain what ossify is and who it's for in plain language.

Content:
- One-paragraph pitch: "If you work with neuron reconstructions that have meshes, skeletons, and synapse tables, ossify lets you move data between them without writing your own mapping code."
- A single compelling figure (the 2D colored cell plot from the quickstart).
- A ~10-line code block showing load → measure → plot.
- Pointers to "Concepts" and "Getting Started" as next steps.
- Brief comparison to alternatives (Navis, MeshParty, CloudVolume) — keep the existing section but tighten it.

---

### Section 2: Core Concepts (`concepts.md`) — new page

**Goal:** Teach the mental model to someone who might know Python and neuroscience basics but hasn't worked with EM reconstruction data at synaptic resolution.

Subsections:

#### 2a. Representations of a Neuron

Explain each representation with its strengths, typical scale, and when you'd use it:

- **Surface mesh** — the full 3D shape from segmentation. Millions of vertices, triangle faces. Captures detailed geometry (surface area, volume) but hard to reason about branching topology.
- **Skeleton** — a rooted tree structure, generally derived from the mesh (but doesn't have to be). Thousands to low tens of thousands of vertices. Good for topology, path analysis, and branch classification. Most natural for analysis.
- **Graph** — the fundamental representation of neuronal topology in EM analysis databases. Roughly 2x skeleton vertices. Like a skeleton but without the tree constraint — can represent cycles or more complex connectivity. First-class in ossify because it makes it easy to bring in the raw database representation, and it's a generic way to embed objects in space with continuous topology without structural assumptions.
- **Point annotations** — sparse locations with metadata (synapses, branch labels, manual annotations). Thousands to tens of thousands per neuron. Don't define shape, but carry critical biological information.
- A figure showing all three overlaid or side-by-side for the same cell.
- Note: you don't need all representations. Ossify works fine with just a skeleton and synapses, for example.

#### 2b. The Linking Problem
- These representations describe the *same physical object* but have different vertex sets.
- Common tasks require combining information across them: "what is the total synapse count on each skeleton branch?" or "what compartment label does each mesh vertex belong to?"
- **Why not just nearest-neighbor?** Neurons have tortuous (winding, complex) structure. Nearest-neighbor is usually right but wrong often enough to be annoying — a synapse on one branch might map to a nearby but topologically distant branch. Ossify uses explicit mappings that fully preserve the correct correspondence.
- Ossify solves this with **Links** — explicit mappings between vertex sets that let you translate features and indices automatically. This makes it easy to aggregate or propagate features across the representations where they're naturally defined (e.g., surface area lives on the mesh but you want it per skeleton vertex).
- Diagram: skeleton vertices ↔ mesh vertices ← synapse points, with arrows showing mapping direction.

#### 2c. The Cell Container
- A `Cell` holds one or more layers (mesh, skeleton, graph) plus annotations (point clouds), all sharing a coordinate space.
- Layers can be linked so features flow between them.
- Masking one layer automatically filters linked layers — e.g., mask the skeleton to dendrite-only and the synapse count updates accordingly.

#### 2d. Compartments and the SWC Standard
- Neurons are typically divided into compartments: soma, axon, dendrite, etc.
- Compartment labels follow the SWC format standard (0=undefined, 1=soma, 2=axon, 3=dendrite, etc.)
- These labels are stored as integer features on skeleton vertices
- (Future: Enum classes like `osy.DENDRITE` for readability)

#### 2e. Where Does the Data Come From?
- Brief intro to CAVE / CAVEclient — the database system hosting MICrONS, FlyWire, etc.
- Ossify can load directly from CAVE, from `.osy` files, or from raw arrays you construct yourself.
- You don't need CAVE access to use ossify — file-based workflows work fine.

---

### Section 3: Getting Started (`getting_started.md`) — rewrite

**Goal:** Hands-on walkthrough using the example cell. The reader should be able to run every code block.

Content:
1. Install ossify (`pip install ossify`)
2. Load the example cell from the hosted URL
3. Inspect it: `cell.describe()`, check layer names, vertex counts
4. Look at the skeleton: root, branch points, end points, cable length
5. Look at annotations: how many pre/post synapses, what features they carry
6. Map a feature: bring mesh `size_nm3` onto skeleton via `map_features_to_layer`
7. Apply a mask: filter to dendrite only, re-measure cable length and synapse count
8. Make a 2D plot colored by compartment
9. Save the cell to a local file

Each step should have 3-5 lines of code and a brief explanation of what's happening and why.

---

### Section 4: User Guide — revised structure

Keep much of the existing content but reorganize for better flow. Each page should open with a "why would I do this?" paragraph, then show examples with the real cell.

#### 4a. The Cell Object (`cell_object.md`) — revise
- Creating cells, adding layers, metadata
- `describe()` for inspection
- Layer access shortcuts (`.s`, `.m`, `.g`, `.a`)
- Keep but tighten existing content

#### 4b. Features and Data on Layers (`shared_layer_features.md`) — revise
- What features are (vertex-level metadata stored in DataFrames)
- Adding, reading, listing features
- Feature names and types

#### 4c. Linking and Mapping (`linking_and_mapping.md`) — new page (replaces parts of working_with_graphs.md)

This is the key differentiator of ossify. Give it its own page.

- What a `Link` is and how it connects layers
- `map_features_to_layer` — moving data (e.g., synapse counts onto skeleton, compartment labels onto mesh)
- Aggregation options: mean, sum, majority
- `map_index_to_layer` — translating vertex identities between layers
- Worked example: "For each skeleton vertex, count how many synapses map to it"

#### 4d. Working with Skeletons (`working_with_skeletons.md`) — revise
- Root, tree structure, parent array
- Branch/end points
- Cable length, depth, topological order, segments
- DAG cache and path queries

#### 4e. Working with Meshes (`working_with_meshes.md`) — revise
- Vertices, faces, surface area
- Trimesh conversion
- When you'd use the mesh vs. the skeleton

#### 4f. Working with Graphs (`working_with_graphs.md`) — revise
- When to use a graph layer (non-tree connectivity)
- Sparse graph representations, path finding, distance computations

#### 4g. Working with Annotations (`working_with_annotations.md`) — revise
- Adding point annotations (synapses, etc.)
- `vertices_from_linkage` — creating annotation points from a link rather than explicit coordinates
- Querying and filtering annotations

#### 4h. Masking and Filtering (`masking_and_filtering.md`) — revise
- `apply_mask` and `mask_context`
- How masks propagate across linked layers
- Example: restrict analysis to a single dendrite branch

#### 4i. Visualization (`visualization_and_plotting.md`) — revise
- `plot_cell_2d`, `plot_morphology_2d`, `plot_cell_multiview`
- Projections (xy, xz, etc.)
- Coloring by feature (discrete palettes, continuous colormaps, colorbar)
- Line width and point size mapping
- Scale bars, figure sizing with `units_per_inch`
- Gallery of example plots

#### 4j. Algorithms (`algorithms_and_analysis.md`) — revise
- Strahler number
- Feature smoothing
- Axon classification (synapse flow, spectral split)
- Synapse betweenness
- Segregation index
- Each with a brief "what it measures" + code example

#### 4k. Data Import and Export (`data_import_export.md`) — revise
- Loading from `.osy` files and cloud paths
- Saving cells
- Loading from CAVE with `load_cell_from_client`
- Importing legacy MeshWork files
- Building a cell from raw arrays (for custom pipelines)

---

### Section 5: API Reference — keep as-is
The auto-generated mkdocstrings reference pages are fine. No changes needed beyond ensuring docstrings are accurate.

### Section 6: FAQ (`faq.md`) — revise
- Keep, but add questions that come from the conceptual framing:
  - "Do I need CAVE access to use ossify?"
  - "What's the difference between a layer and an annotation?"
  - "How do I know which layer to use as source vs target in mapping?"

### Section 7: Changelog — keep as-is

---

## Proposed `nav` in mkdocs.yml

```yaml
nav:
  - Home: index.md
  - Core Concepts: concepts.md
  - Getting Started: getting_started.md
  - User Guide:
      - The Cell Object: cell_object.md
      - Features and Data: shared_layer_features.md
      - Linking and Mapping: linking_and_mapping.md
      - Working with Skeletons: working_with_skeletons.md
      - Working with Meshes: working_with_meshes.md
      - Working with Graphs: working_with_graphs.md
      - Working with Annotations: working_with_annotations.md
      - Masking and Filtering: masking_and_filtering.md
      - Visualization: visualization_and_plotting.md
      - Algorithms: algorithms_and_analysis.md
      - Data Import and Export: data_import_export.md
      - FAQ: faq.md
  - API Reference:
      - Overview: reference/index.md
      - Core Classes: reference/core.md
      - Data Layers: reference/layers.md
      - Algorithms: reference/algorithms.md
      - Plotting: reference/plotting.md
      - File I/O: reference/io.md
      - External Integrations: reference/external.md
      - Complete API: reference/api.md
  - Changelog: changelog.md
```

Key changes: "Core Concepts" and "Getting Started" promoted to top-level nav tabs (not buried under User Guide). New "Linking and Mapping" page in User Guide.

---

## Implementation Order

1. **`concepts.md`** — new, highest priority. This is the missing foundation.
2. **`index.md`** — rewrite to be concise and compelling.
3. **`getting_started.md`** — rewrite as hands-on tutorial with real data.
4. **`linking_and_mapping.md`** — new page for ossify's core differentiator.
5. **Revise existing User Guide pages** — add "why" intros, use real cell examples, remove AI boilerplate warnings.
6. **Update `mkdocs.yml`** nav structure.
7. **FAQ updates** — add conceptual questions.

## Figures Needed

- Conceptual diagram: mesh vs skeleton vs annotations for the same neuron (could be a multi-panel 2D plot)
- Linking diagram: showing how vertices map between layers (could be a simple schematic)
- The compartment-colored cell plot (already exists as `quickstart_img.png`)
- Example masking before/after
- Algorithm output examples (e.g., Strahler coloring)

Many of these can be generated with ossify's own plotting tools using the example cell.
