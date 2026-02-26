"""Spatial-temporal dataset middle interface.

Extends :class:`~liulian.data.ts.timeseriesdataset.TimeSeriesDataset` with
graph / spatial structure integration.  Provides multiple strategies for
incorporating structural information into model inputs:

* ``graph_mode='edge_index'`` — store a sparse ``(2, E)`` edge-index array
  alongside the temporal data, passed to GNN convolutions at training time.
  This follows the pattern used by ``STGNNSequence*Dataset`` in the
  reference project where ``edges`` are handed to the training loop.
* ``graph_mode='adj_matrix'`` — dense adjacency matrix ``(N, N)`` for
  models that require it (e.g. spectral GNNs, Diffusion-Conv).
* ``graph_mode='graphlet_features'`` — append neighbour predictions /
  features from a *graphlet* (local sub-graph) to each station's input
  vector.  This mirrors the *graphlet* training in the reference project
  (``train_graphlet``, ``train_transformer_graphlet``).
* ``graph_mode='none'`` — no graph information.

Adapted from:
- refer_projects/swiss-river-network-benchmark/swissrivernetwork/benchmark/dataset.py
  (STGNNSequenceFullDataset, STGNNSequenceWindowedDataset)
- refer_projects/swiss-river-network-benchmark/swissrivernetwork/benchmark/train_single_model.py
  (train_stgnn, train_transformer_stgnn)
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from liulian.data.spec import FieldSpec, TopologySpec
from liulian.data.ts.timeseriesdataset import TimeSeriesDataset, TimeSeriesSplit


class SpatialTempoDataset(TimeSeriesDataset):
    """Middle interface adding spatial / graph structure to time-series data.

    All :class:`TimeSeriesDataset` parameters are forwarded.  The extra
    parameters control how graph topology is exposed to models.

    Parameters
    ----------
    graph_mode : str
        ``'none'`` | ``'edge_index'`` | ``'adj_matrix'`` |
        ``'graphlet_features'``.
    graph_metadata : dict | None
        Arbitrary metadata about the graph (e.g. ``graph_name``).
    """

    domain: str = 'spatiotemporal'
    version: str = '1.0'

    def __init__(
        self,
        *,
        graph_mode: str = 'none',
        graph_metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.graph_mode = graph_mode
        self.graph_metadata = graph_metadata or {}

        # Pre-compute graph representations
        self._edge_index: Any | None = None
        self._adj_matrix: Any | None = None

        if self.topology is not None:
            self._build_graph_arrays()

    # ------------------------------------------------------------------
    # Graph array construction
    # ------------------------------------------------------------------

    def _build_graph_arrays(self) -> None:
        """Pre-compute ``_edge_index`` and ``_adj_matrix`` from topology.

        Uses :attr:`self.B` so the arrays are in the active backend
        format (numpy or torch).
        """
        if self.topology is None:
            return

        node_ids = self.topology.node_ids
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)

        # Edge index — (2, E)
        src_list, dst_list = [], []
        for s, d in self.topology.edges:
            if s in id_to_idx and d in id_to_idx:
                src_list.append(id_to_idx[s])
                dst_list.append(id_to_idx[d])
        if src_list:
            self._edge_index = self.B.asarray(
                [src_list, dst_list], dtype='int64',
            )
        else:
            self._edge_index = self.B.empty((2, 0), dtype='int64')

        # Adjacency matrix — (N, N)
        adj = self.B.zeros((n, n), dtype='float32')
        for s, d in zip(src_list, dst_list):
            adj[s, d] = 1.0
        self._adj_matrix = adj

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def edge_index(self) -> Any | None:
        """Sparse edge-index array ``(2, E)`` or ``None``."""
        return self._edge_index

    @property
    def adj_matrix(self) -> Any | None:
        """Dense adjacency matrix ``(N, N)`` or ``None``."""
        return self._adj_matrix

    @property
    def num_nodes(self) -> int:
        if self.topology is not None:
            return self.topology.num_nodes
        return len(self.station_ids) if self.station_ids else 0

    @property
    def node_coordinates(self) -> Any | None:
        """Return ``(N, 2)`` coordinate array or ``None``."""
        if self.topology is None or not self.topology.coordinates:
            return None
        coords = []
        for nid in self.topology.node_ids:
            coords.append(
                list(self.topology.coordinates.get(nid, (0.0, 0.0)))
            )
        return self.B.asarray(coords, dtype='float32')

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        out = super().info()
        out.update(
            {
                'graph_mode': self.graph_mode,
                'num_nodes': self.num_nodes,
                'num_edges': (
                    self._edge_index.shape[1] if self._edge_index is not None else 0
                ),
                'graph_metadata': self.graph_metadata,
            }
        )
        return out
