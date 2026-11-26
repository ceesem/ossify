"""Lasso skeleton vertices on a feature embedding and label them on the cell.

A Jupyter widget (anywidget) for building compartment labels by eye. See
``vertex.py`` for the details and ``../label_compartments.ipynb`` for a worked example.
"""

from .vertex import LABELS, VertexExplorer, read_labels

__all__ = ["LABELS", "VertexExplorer", "read_labels"]
