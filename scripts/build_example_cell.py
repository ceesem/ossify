"""Build the distributable four-layer example cell.

The source mesh is 2.6M vertices. Quadric decimation (fast-simplification via
trimesh) does most of the work and preserves surface area well, but it floors
around 634k faces because the reconstruction is in 970 disconnected bodies --
65 of which are over 1000 vertices, so they are real morphology, not dust.
Vertex clustering takes it the rest of the way.

Note trimesh's `percent` is the fraction to REMOVE, not to keep.
"""

import sys
import numpy as np, pandas as pd, trimesh, ossify
from scipy.spatial import cKDTree
from ossify import Cell, Link

GRID = float(sys.argv[1]) if len(sys.argv) > 1 else 700.0
OUT = sys.argv[2] if len(sys.argv) > 2 else "864691135336055529_full.osy"
SRC = "tests/data/test_cell_with_mesh.osy"

src = ossify.load_cell(SRC)
V0, F0 = src.mesh.vertices, src.mesh.faces_positional
src_area = src.mesh.surface_area()

dec = trimesh.Trimesh(vertices=V0, faces=F0, process=False).simplify_quadric_decimation(
    percent=0.99
)
V, F = np.asarray(dec.vertices), np.asarray(dec.faces)

keys = np.floor(V / GRID).astype(np.int64)
_, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
n_cells = len(counts)
reps = np.zeros((n_cells, 3))
for axis in range(3):
    reps[:, axis] = np.bincount(inverse, weights=V[:, axis], minlength=n_cells)
reps /= counts[:, None]

nf = inverse[F]
keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
nf = np.unique(np.sort(nf[keep], axis=1), axis=0)
used = np.unique(nf)
remap = np.full(n_cells, -1, dtype=np.int64)
remap[used] = np.arange(len(used))
mesh_verts, mesh_faces = reps[used], remap[nf]

# Link each new vertex to the graph, via the nearest ORIGINAL vertex that
# actually has a link -- 502 source vertices carry the unmapped sentinel 0,
# and inheriting one of those would put a non-existent graph id in the link.
m2g = np.asarray(src.mesh.map_index_to_layer("graph"))
valid = np.isin(m2g, src.graph.vertex_index)
_, nearest_valid = cKDTree(V0[valid]).query(mesh_verts)
new_m2g = m2g[valid][nearest_valid]
assert np.isin(new_m2g, src.graph.vertex_index).all()

out = Cell(name=src.name, meta=dict(src.meta or {}))
out.add_skeleton(
    vertices=src.skeleton.nodes[src.skeleton.spatial_columns],
    edges=src.skeleton.edges,
    spatial_columns=src.skeleton.spatial_columns,
    root=src.skeleton.root,
    features=src.skeleton.features,
    edges_as_positional=False,
)
out.add_graph(
    vertices=src.graph.nodes[src.graph.spatial_columns],
    edges=src.graph.edges,
    spatial_columns=src.graph.spatial_columns,
    features=src.graph.features,
    edges_as_positional=False,
    linkage=Link(
        mapping=np.asarray(src.graph.map_index_to_layer("skeleton")),
        target="skeleton",
        map_value_is_index=True,
    ),
)
out.add_mesh(
    vertices=pd.DataFrame(mesh_verts, columns=src.mesh.spatial_columns),
    faces=mesh_faces,
    spatial_columns=src.mesh.spatial_columns,
    faces_as_positional=True,
    linkage=Link(mapping=new_m2g, target="graph", map_value_is_index=True),
)
for anno in ("pre_syn", "post_syn"):
    a = src.annotations[anno]
    out.add_point_annotations(
        name=anno,
        vertices=a.nodes,
        spatial_columns=a.spatial_columns,
        linkage=Link(
            mapping=np.asarray(a.map_index_to_layer("graph")),
            target="graph",
            map_value_is_index=True,
        ),
    )
ossify.save_cell(out, OUT, allow_overwrite=True)
import os

print(
    f"grid={GRID:.0f}nm  mesh {out.mesh.n_vertices:,} v / {len(out.mesh.faces):,} f  "
    f"area {100 * out.mesh.surface_area() / src_area:.0f}%  "
    f"file {os.path.getsize(OUT) / 1e6:.1f}MB"
)
