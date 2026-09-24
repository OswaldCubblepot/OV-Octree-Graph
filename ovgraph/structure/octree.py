"""
Adaptive octree for instance occupancy representation.

In the paper "Open-Vocabulary Octree-Graph for 3D Scene Understanding",
each 3D instance is represented by an *adaptive octree*:

  - adaptive initialization: the root cube is initialized from the object's
    shape (the root is the tight bounding cube of the instance points, so
    objects with a large aspect ratio, e.g. walls and floors, are described
    compactly);
  - adaptive subdivision: a cell is subdivided only when it contains
    instance points, until ``min_cell_size`` / ``max_depth`` is reached;
  - the set of occupied leaf cells encodes the instance occupation with far
    fewer primitives than the raw point set (e.g. tens of KB vs. millions of
    points), which supports efficient occupancy query and path planning.
"""

import numpy as np
from scipy.spatial import cKDTree

_EPS = 1e-9


class AdaptiveOctree:
    """Occupancy octree of one 3D instance.

    Attributes:
        root: (cx, cy, cz, half_size) of the root cube.
        cells: (N, 4) float array of occupied leaf cells (cx, cy, cz, half_size).
        counts: (N,) int array, number of instance points inside each cell.
    """

    def __init__(self, pts=None, min_cell_size=0.05, max_depth=8):
        self.min_cell_size = float(min_cell_size)
        self.max_depth = int(max_depth)
        self.root = None
        self.cells = np.empty((0, 4), dtype=np.float64)
        self.counts = np.empty((0,), dtype=np.int64)
        self._ball_tree = None
        if pts is not None and len(pts) > 0:
            self.build(pts)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def build(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 3, "pts must be (N, 3)"
        assert len(pts) > 0, "cannot build an octree from empty points"

        # -------------------------------------------------------------- #
        # adaptive initialization: root cube from the object's shape
        # -------------------------------------------------------------- #
        xyz_min = pts.min(axis=0)
        xyz_max = pts.max(axis=0)
        center = 0.5 * (xyz_min + xyz_max)
        # aspect-ratio aware: root half size = half of the longest extent,
        # so flat objects are not over-tessellated.
        half = 0.5 * (xyz_max - xyz_min).max() + _EPS
        self.root = np.array([*center, half], dtype=np.float64)

        # -------------------------------------------------------------- #
        # adaptive subdivision (stack based, vectorized per node)
        # -------------------------------------------------------------- #
        cells, counts = [], []
        stack = [(center, half, 0, pts)]
        while stack:
            c, h, depth, p = stack.pop()
            n = len(p)
            if n == 0:
                continue
            if h <= self.min_cell_size or depth >= self.max_depth:
                cells.append([*c, h])
                counts.append(n)
                continue

            # split the 8 octants
            ch = 0.5 * h
            for dz in (-ch, ch):
                for dy in (-ch, ch):
                    for dx in (-ch, ch):
                        cc = np.array([c[0] + dx, c[1] + dy, c[2] + dz])
                        m = np.all(np.abs(p - cc) <= ch, axis=1)
                        if m.any():
                            stack.append((cc, ch, depth + 1, p[m]))

        self.cells = np.array(cells, dtype=np.float64)
        self.counts = np.array(counts, dtype=np.int64)
        self._ball_tree = None
        return self

    # ------------------------------------------------------------------ #
    # occupancy query
    # ------------------------------------------------------------------ #
    def _tree(self):
        if self._ball_tree is None and len(self.cells) > 0:
            self._ball_tree = cKDTree(self.cells[:, :3])
        return self._ball_tree

    def query_occupancy_batch(self, pts):
        """Return a boolean array, True where ``pts`` falls inside an occupied cell."""
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]
        out = np.zeros(len(pts), dtype=bool)
        if len(self.cells) == 0:
            return out

        tree = self._tree()
        # L2 radius from a cell center to a corner is half * sqrt(3); query a
        # bit more, then verify exactly with the infinity norm.
        radius = float(self.cells[:, 3].max()) * np.sqrt(3.0) + _EPS
        query_radius = getattr(tree, "query_radius", None) or getattr(tree, "query_ball_point")
        inds = query_radius(pts, r=radius)

        # chunked exact verification (inf-norm cube test)
        chunk = 1 << 16
        for s in range(0, len(pts), chunk):
            e = min(s + chunk, len(pts))
            for i in range(s, e):
                j = inds[i]
                if len(j) == 0:
                    continue
                d = np.abs(pts[i] - self.cells[j, :3])
                out[i] = np.any(np.all(d <= self.cells[j, 3, None], axis=1))
        return out

    def query_occupancy(self, pt):
        return bool(self.query_occupancy_batch(pt)[0])

    # ------------------------------------------------------------------ #
    # voxelization helper (used by path planning)
    # ------------------------------------------------------------------ #
    def mark_grid(self, voxel_centers, grid_shape, grid):
        """Mark grid voxels whose center lies inside an occupied cell.

        Args:
            voxel_centers: (G, 3) centers of all grid voxels.
            grid_shape: (nx, ny, nz) of the grid.
            grid: int32 array of shape grid_shape, modified in place
                (1 = occupied).
        """
        if len(self.cells) == 0:
            return grid
        occ = self.query_occupancy_batch(voxel_centers)
        grid.reshape(-1)[occ] = 1
        return grid

    # ------------------------------------------------------------------ #
    # stats & io
    # ------------------------------------------------------------------ #
    def n_cells(self):
        return len(self.cells)

    def n_points(self):
        return int(self.counts.sum()) if len(self.counts) else 0

    def bytes_estimate(self):
        """Rough serialized size: 4 float64 per cell."""
        return len(self.cells) * 4 * 8

    def state_dict(self):
        return {
            "min_cell_size": self.min_cell_size,
            "max_depth": self.max_depth,
            "root": self.root,
            "cells": self.cells,
            "counts": self.counts,
        }

    @classmethod
    def from_state_dict(cls, d):
        obj = cls(min_cell_size=d["min_cell_size"], max_depth=d["max_depth"])
        obj.root = d["root"]
        obj.cells = d["cells"]
        obj.counts = d["counts"]
        return obj

    def __len__(self):
        return len(self.cells)
