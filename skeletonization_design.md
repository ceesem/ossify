# Skeletonization: design decisions

Written 2026-11-24, at the end of the session that added graph skeletonization to
ossify. It records what was built, why each choice was made, the evidence behind
the thresholds, and what is still open. Read this before changing any of it.

## What exists

| Piece | Where | What it does |
|---|---|---|
| `skeletonize(cell, ...)` | `src/ossify/skeletonization.py` | Skeletonize a cell's graph layer, add the skeleton, link it to the graph |
| `skeletonize_graph(vertices, edges, ...)` | same | The same on plain arrays; returns a `SkeletonResult` |
| `contract_vertices(...)` | same | Merge a set of graph vertices into one new vertex, tracking the map |
| `detect_soma(...)` | same | Which graph vertices are soma, from inscribed radius and volume |
| `find_soma_point(...)` | same | Guess the soma centre from the graph, or None |
| `add_skeleton_radius(cell)` | same | Add `radius` and `radius_local` skeleton features |
| `method="pcg_skel"` | `src/ossify/_pcg_skel_compat.py` | Reproduce the CAVE SkeletonService skeleton |
| `load_cell_from_l2graph(root_id, client)` | `src/ossify/translate.py` | Fetch the L2 graph from CAVE and skeletonize locally |

Tests: `tests/test_skeletonization.py`, `tests/test_load_from_l2graph.py`.
Reference docs: `docs/reference/algorithms.md`, `docs/reference/external.md`.

The original port target was `meshparty/skeletonize.py` (via `pcg_skel`). The
requirements were: graph-to-skeleton mapping, rooting, a soma collapse, and the
ability to add a root vertex that is not in the graph.

## Core design

### One primitive: contraction with a tracked map

Soma collapse, adding a vertex that is not in the graph, and the
graph-to-skeleton map are the same operation: a quotient of the graph. So the
soma is contracted **in the graph, before skeletonizing**, and TEASAR is rooted
at the contracted vertex. MeshParty collapsed the soma on the finished skeleton,
which needed fiddly surgery and wasted effort skeletonizing the soma surface.

After a contraction, edges into the contracted vertex **keep their original
graph length**, not the distance to the new vertex's position. Graph distance
from the root is then distance from the soma *surface*. Measured from the
centre, every vertex just outside the soma is at least a soma radius away,
beyond the reach of invalidation, and each becomes a stub branch. (Found on
v1dd 864691132806671718: 19 branches off the soma, 7 of them stubs.)

### Path extraction (the default method)

1. Dijkstra from the root once. Sort vertices by distance; no argmax per path.
2. Take the farthest still-valid vertex and walk toward the root.
3. **While in uncovered territory, follow the shortest route to the root. On
   reaching a vertex an earlier path already covered, turn and take the
   shortest route to that path.**
4. Invalidate everything within `invalidation` (graph distance) of the new
   path, *including the vertex it joined*, so the root covers its surroundings.

Step 3 is a fix, not a nicety. Without it a thick process split lengthwise by a
chunk boundary (two parallel chains of L2 vertices, a "ladder") gets one
skeleton path per chain, because branches beyond it each have an equally short
route home down a different chain. On minnie 864691136578820884 this doubled
58.9 um of apical dendrite. 11 of 80 sampled cells were affected; 2 remain with
at most 8.3 um flagged, not investigated (they may be real loops).

### The graph-to-skeleton map

One multi-source Dijkstra from all skeleton vertices at the end: each graph
vertex goes to its geodesically nearest skeleton vertex. MeshParty accumulated
the map during invalidation, which gives the nearest vertex among the paths
that happened to invalidate it, not the nearest overall.

### Disconnected graphs

Raise. A cell's graph is assumed to be one component.

### Single-vertex graphs

Return a one-vertex skeleton (default method). `pcg_skel` still raises.

## Soma handling

Three ways to collapse a soma, in `skeletonize_graph` / `skeletonize`:

- `soma_radius`: a ball around `soma_point`, restricted to vertices connected
  to the soma without leaving the ball (so a process passing back through is
  not short-circuited).
- `soma_mask`: an explicit vertex set, used as given.
- `vertex_radius` + `vertex_volume` (cell level: `detect_soma=True`): find the
  soma from the graph. **This is the recommended one.**

`soma_point` alone just roots at the nearest graph vertex.

### Why not a ball

Studied on 80 cells (40 minnie65_phase3_v1, 40 v1dd; `proofreading_status_and_strategy`
with axon and dendrite both proofread, exactly one nucleus). "Stub" = a branch
off the root with under 20 um of cable. "Long branch" = 50 um or more.

| Rule (minnie / v1dd) | Vertices merged, median | Cells with a stub | Long branches, mean |
|---|---|---|---|
| Ball 5 um | 48 / 34 | 35% / 30% | 6.1 / 6.3 |
| Ball 7.5 um | 83 / 68 | 18% / 25% | 7.0 / 7.9 |
| Ball 15 um | 137 / 110 | 0% / 12% | 9.2 / 10.4 |
| Service method | 96 / 77 | 28% / 8% | 6.8 / 7.6 |
| **Adopted rule** | 43 / 37 | 0% / 0% | 6.2 / 7.2 |

A small ball leaves soma-surface stubs; a large one splits dendrites that fork
inside it into separate primaries and merges far more of the graph. (Numbers in
the first three rows predate the ladder fix; the direction did not change.)

### The adopted rule: `detect_soma`

Chosen by Casey after comparing rules against meshes in Neuroglancer:
"wider core, two pass, split thick and near bases".

**Seed.** The largest inscribed sphere within 15 um of the soma point, or the
nearest vertex if there is none. The largest sphere within 15 um of the nucleus
was also the cell's largest in all 80 cells. Do not seed at the nearest vertex:
in about 10% of cells it is not part of the soma blob.

**First pass, the core.** Vertices join when either
- they share a graph edge shorter than the larger of their two radii (one
  centre lies inside the other's inscribed sphere, "engulfment"), or
- both spheres are at least 0.5x the seed's and they overlap ("wider core").

Neurite vertices satisfy neither: they are about 1.5-1.8 um apart and 0.14-0.19
um thick at the median. The overlap clause exists because chunk boundaries cut
a soma into several large pieces, none big enough to engulf the next (v1dd
864691132710501931: core 16 -> 65 vertices, soma radius 3.8 -> 6.1 um). With
it, cells with skeleton running through a large sphere went 15% -> 2% (minnie)
and 32% -> 0% (v1dd) with no extra creep.

**Second pass, branches.** Skeletonize from the core, then for each branch off
the root:
- under `min_branch_length` (20 um) of cable: it is leftover soma surface, not
  a process. Absorb it.
- it forks into two branches that each pass that test, the stretch before the
  fork is thick (median calibre >= 0.25x the seed's radius), and the fork is
  near (within 1.5 soma radii of the soma point, soma radius = sphere-equivalent
  of the core volume): the stretch is a bulge of the soma. Absorb it so the two
  branches leave the root separately.

Repeat (re-skeletonizing) until nothing is absorbed; capped at 8 rounds, settles
in 2-3.

### Rejected along the way

- **Engulfment at 1.5x radius**: reaches 26.8 um from the nucleus in one cell.
- **"Thick core" cap** (only spheres >= 0.5 peak may engulf): barely changes
  anything; not worth a parameter.
- **Wider core at 0.33x peak**: core extent grows from ~11 to ~14 um.
- **Coverage / root reach** (pre-invalidating vertices around the root so they
  cannot become tips): worked for stubs but was a hidden radius. Replaced by
  the second pass, whose threshold is a branch length with a plain meaning.
- **Splitting on thickness alone**: creeps up thick apical trunks (24 um).
  The "near" condition bounds it.
- **A radius "knee" to find where a branch starts**: the profile tapers over
  8-15 um with no sharp neck.

### Known soft spots

- Worst-case merged vertex is 21 um from the nucleus in one v1dd cell. Believed
  to be a short twig hanging off an absorbed base; not confirmed.
- A real process with under 20 um of cable off the soma is absorbed.
- If no branch reaches 20 um, everything is soma and the skeleton is one vertex.
- Missing radius/volume values count as zero.

### Guessing the soma: `find_soma_point`

Used when no soma point is given or found. Seed at the global largest sphere,
build the core, return its volume-weighted centroid if
- largest sphere >= 2.2 um (`min_peak_radius`), and
- core volume >= that of a 3 um sphere (`min_soma_radius`).

Evidence: somas had largest sphere >= 2.7 um and core radius >= 4.0 um; nothing
more than 25 um from a nucleus had a sphere over 1.7 um. The sphere test is what
separates them: one minnie cell had a non-soma blob with an 8.3 um equivalent
radius. The guess was a median 1.2 um, at most 2.7 um, from the nucleus point.
Only cortical neurons were checked. A multi-soma object yields the largest.

Casey's position: keep using the nucleus lookup when there is one; the guess is
a fallback.

## Radius

`add_skeleton_radius` adds:

- `radius`: `pcg_skel`'s per-segment equivalent radius,
  `sqrt(volume / (pi * length))`, with the segment containing the root treated
  as a sphere. Matches the service's values to 1e-7 on its own skeleton for the
  reference cell. Quirk kept for fidelity: if the root is not a branch point,
  its "segment" is a whole process and gets a sphere's radius.
- `radius_local`: the largest `max_dt_nm` among graph vertices mapping to each
  skeleton vertex. Reads low where a chunk boundary splits a thick process
  lengthwise, because each piece only sees part of the cross-section.

Not built: a split-invariant local radius. Two candidates were identified:
- `size_nm3 / area_nm2` pooled over neighbours. `area_nm2` counts only external
  faces (confirmed by Casey), so sum(V)/sum(A) is unaffected by the split and
  can be pooled in any direction. `2V/A` is a stable ~0.56x of `max_dt_nm`
  (IQR 0.49-0.62) on the test cell, so one constant calibrates it.
- Summing `max_dt` across the split: exact for a round cross-section cut by a
  chord, but needs side-by-side neighbours identified.

## SkeletonService compatibility (`method="pcg_skel"`)

The service calls `pcg_skel.pcg_meshwork`, which calls MeshParty's
`skeletonize_mesh` with `cc_vertex_thresh=0`, `remove_zero_length_edges=True`,
sphere collapse, no smoothing. `_pcg_skel_compat.py` follows that path step by
step. On the test cell (`tests/data/skel.json`, a version 4 service skeleton)
it gives identical vertices, edges (same order), root and map.

What it keeps that the default does differently, because matching is the point:
float32 edge weights with repeated edges summed; growth from a graph extremity
even when a soma is given; branches attached at the nearest skeleton vertex by
a bounded search from the tip; the map taken from invalidation; soma collapse
on the finished skeleton taking vertices inside the sphere **plus one edge
beyond it**; zero-length skeleton edges merged.

Deliberate deviations (where the original breaks): runs of three or more
coincident skeleton vertices are merged fully; a bounded search that misses
through round-off is retried unbounded; disconnected graphs raise; `root` and
`soma_mask` are rejected.

Casey considers many of the service's decisions wrong; this exists only for
compatibility.

## The CAVE loader

`load_cell_from_l2graph` fetches the L2 graph, then the L2 table in concurrent
chunks, resolves the soma, skeletonizes, and optionally adds radius, synapses
and compartments. **Every step after the graph is an option** (Casey's
requirement), and each is its own helper:

| Step | Option | Default |
|---|---|---|
| Build a skeleton | `build_skeleton` | on |
| Nucleus lookup by `pt_root_id` | `lookup_soma`, `soma_table` | on |
| Else guess from the graph | `guess_soma` | on |
| Collapse the detected soma | `detect_soma` | on |
| `radius`, `radius_local` | `skeleton_radius` | on |
| `compartment` (needs synapses) | `compartments` | on |
| Other `skeletonize` arguments | `skeleton_kwargs` | none |

Decisions:
- Soma order: given, then nucleus lookup, then graph guess.
  `cell.meta["soma_source"]` records which.
- Several nuclei found: warn, then fall back to the graph guess. (Changed late;
  it previously built without a soma. Not explicitly confirmed by Casey.)
- Missing L2 positions are filled with the mean of placed neighbours, as
  `pcg_skel` does; raise if any cannot be placed.
- `compartment` uses `label_axon_from_synapse_flow` with the service's settings
  (segregation threshold 0.5, extended to segments), root marked soma. Not
  compared against the service's labels.
- The default `soma_table` is the datastack's. For minnie65_phase3_v1 that is
  `nucleus_neuron_svm`, not `nucleus_detection_v0`; pass `soma_table` to get
  the service's tables (`nucleus_detection_v0` + `nucleus_alternative_points`).
- `load_cell_from_client` (service-based) was left untouched.

### Fetch timing

Compute is under 1% of wall time; the CAVE calls are the rest.

- Default skeletonization: 10-90 ms for 4k-24k L2 vertices. `detect_soma`
  adds roughly one more skeletonization per round (~75 ms per cell average).
- `level2_chunk_graph`: ~3-10 s. `get_leaves(stop_layer=2)` is about as slow
  (3.8 s vs 5.3 s), so fetching ids separately to parallelize graph and table
  saves only about a second. Not done.
- L2 table: one request ~8.7 s per 10k ids; eight concurrent chunks ~2.1 s per
  10k. Adopted. All of the time is server-side wait before the first byte,
  ~0.7 ms per id, linear; the payload is tiny.
- Likely cause (unconfirmed): per-id Python loops in PCGL2Cache's
  `handle_attributes` (cell unpacking, `_add_offset_to_coords`, volume rescale,
  JSON). Needs profiling on the server. Casey agrees there is room there.

## Not done / open

1. **Cycle audit.** TEASAR silently cuts loops. Plan: project each graph edge
   through the map; an edge whose endpoints are far apart in the tree witnesses
   a loop of about that length. Only loops many times longer than the local
   radius matter (Casey), and benign graphs produce plenty of short ones: one
   test cell has ~4,460 independent cycles from chunking alone. Report, do not
   fix; keep `SkeletonLayer` a tree.

   Tried 2026-11-25 on all 3,604 axon-proofread cells (minnie65 v1979, v1dd
   v1643), in scratch scripts, not in the package. Graph edges whose ends map
   to skeleton vertices more than 30 um apart along the tree, grouped within
   15 um: 645 loops in 528 cells, 505 of them open (arms more than 5 um apart
   for most of their length). Scale separation is clean: about 1,340
   loop-closing edges per cell, about 1,080 far ones in total. The tree cuts a
   loop where both ways round are equally far from the root (shorter arm a
   median 44% of the loop), not at the contact.

   Casey reviewed loops in Neuroglancer: there is no one cue (thinnest point
   or otherwise) that says where the contact is, so do not move the cut
   automatically. The audit stays report-only; fixing is a manual split.
2. **Adaptive invalidation.** `scale * local_radius + const`, radius as a
   per-vertex array. scipy's `limit` is scalar; options are quantized radius
   levels or a virtual source node. Needs the split-invariant radius first.
3. **Post-hoc soma collapse** on a skeleton that already exists (for example a
   service skeleton). The contraction primitive supports it; not wired up.
4. **Should `detect_soma` be the default** in `skeletonize` when the features
   are present? Asked, not answered.
5. **Validate compartment labels** against the service.
6. **Non-neuronal and non-cortical cells** for every threshold above.
7. **L2 cache server speed-up** (PCGL2Cache), outside this repo.

## Where the evidence lives

The study scripts and fetched cells were in a session scratch directory and are
**not** in the repo, so the numbers above cannot be regenerated without
rewriting them. The interactive comparison (14 cells, each soma rule, 3D, with
Neuroglancer links) is at
https://claude.ai/artifact/TA7ZnoaPfPcNxGBqY79RVS (private to Casey).

Cells worth revisiting:
- minnie 864691136578820884: the doubled apical path.
- v1dd 864691132710501931: soma cut into pieces; motivated the wider core.
- v1dd 864691132806671718: refused by the service; soma larger than a 7.5 um ball.
