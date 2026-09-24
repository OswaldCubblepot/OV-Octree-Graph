"""
Octree-graph: an efficient and compact 3D scene representation.

Each node corresponds to one 3D instance and stores its semantics
(visual feature, class name), center and an adaptive octree describing its
occupation. Each edge encodes the spatial relation between two instances:
a semantic direction word (e.g. "right of", "above"), the spatial distance
and the 3D offset vector.

On top of this representation we provide three downstream tasks:
  - instance retrieval: text / feature query -> top-k instances;
  - occupancy query: which instance occupies a 3D position;
  - path planning: A* search on the free voxels of the scene occupation.
"""

import heapq

import networkx as nx
import numpy as np

from .octree import AdaptiveOctree

_EPS = 1e-9

# default 2D heading ("front" direction) in the horizontal plane
DEFAULT_HEADING = np.array([1.0, 0.0])


def _direction_word(vec, heading=None):
    """Return a semantic spatial relation word for the offset ``vec``.

    The vertical relation is checked first ("above"/"below", world +z is up);
    otherwise the horizontal offset is partitioned into four 90-degree
    sectors around the ``heading`` ("front", "left", "behind", "right").
    """
    heading = DEFAULT_HEADING if heading is None else np.asarray(heading, dtype=np.float64)
    heading = heading / (np.linalg.norm(heading) + _EPS)

    dx, dy, dz = vec
    if abs(dz) >= max(abs(dx), abs(dy)):
        return "above" if dz > 0 else "below"

    # rotate the horizontal offset into the heading frame
    cos_t = heading[0]
    sin_t = -heading[1]
    u = dx * cos_t - dy * sin_t   # forward component
    v = dx * sin_t + dy * cos_t   # right component
    ang = np.arctan2(v, u)        # 0 = front, pi/2 = right
    if -np.pi / 4 <= ang < np.pi / 4:
        return "in front of"
    elif np.pi / 4 <= ang < 3 * np.pi / 4:
        return "right of"
    elif -3 * np.pi / 4 <= ang < -np.pi / 4:
        return "left of"
    else:
        return "behind"


class OctreeGraph(nx.Graph):
    """A graph of instances; each node owns an adaptive occupancy octree."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, scene_graph, pcd_xyz, min_cell_size=0.05, max_depth=8,
              edge_dist=1.0, heading=None):
        """Build an octree-graph from a scene graph.

        Args:
            scene_graph: networkx graph produced by ``obtain_3d_segment`` /
                ``graph_back_projcection`` (nodes own 'pt_indices',
                'center', features and optionally 'top1_cls' / 'top5_vocabs').
            pcd_xyz: (N, 3) scene point coordinates.
            min_cell_size / max_depth: octree subdivision parameters.
            edge_dist: maximum center distance between connected instances.
            heading: 2D "front" direction used for relation words.

        Returns:
            OctreeGraph
        """
        G = cls()
        pcd_xyz = np.asarray(pcd_xyz, dtype=np.float64)
        heading = DEFAULT_HEADING if heading is None else heading
        G.graph.update({
            "scan": scene_graph.graph.get("scan", None),
            "n_pts": int(len(pcd_xyz)),
            "min_cell_size": float(min_cell_size),
            "max_depth": int(max_depth),
            "edge_dist": float(edge_dist),
            "heading": np.asarray(heading, dtype=np.float64).tolist(),
        })

        # ---- nodes --------------------------------------------------- #
        node_ids = list(scene_graph.nodes)
        centers = []
        for node in node_ids:
            attrs = scene_graph.nodes[node]
            pt_indices = np.asarray(attrs["pt_indices"], dtype=np.int64).reshape(-1)
            pts = pcd_xyz[pt_indices]
            center = np.asarray(attrs.get("center", np.mean(pts, axis=0)), dtype=np.float64)

            feature = attrs.get("back_prj_feat_mean", attrs.get("feature", None))
            feature = np.asarray(feature, dtype=np.float64) if feature is not None else None
            cap_feature = attrs.get("back_prj_cap_feat_mean", None)
            cap_feature = np.asarray(cap_feature, dtype=np.float64) if cap_feature is not None else None

            octree = AdaptiveOctree(pts, min_cell_size=min_cell_size, max_depth=max_depth)
            G.add_node(node, **{
                "center": center,
                "pt_indices": pt_indices,
                "feature": feature,
                "cap_feature": cap_feature,
                "top1_cls": attrs.get("top1_cls", None),
                "top5_vocabs": attrs.get("top5_vocabs", None),
                "octree": octree,
                "n_pts": int(len(pts)),
            })
            centers.append(center)
        centers = np.stack(centers) if centers else np.empty((0, 3))

        # ---- edges --------------------------------------------------- #
        if len(centers) > 1:
            dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            iu, ju = np.triu_indices(len(centers), k=1)
            for i, j in zip(iu, ju):
                if dist[i, j] > edge_dist:
                    continue
                vec = centers[j] - centers[i]
                G.add_edge(
                    node_ids[i], node_ids[j],
                    vec3d=vec,
                    distance=float(dist[i, j]),
                    relation=f"{node_ids[j]} is {_direction_word(vec, heading)} {node_ids[i]}",
                )
        return G

    # ------------------------------------------------------------------ #
    # downstream: retrieval
    # ------------------------------------------------------------------ #
    def query_by_feature(self, query_feat, top_k=5, use_caption=False):
        """Retrieve the top-k instances closest to a query feature vector."""
        query_feat = np.asarray(query_feat, dtype=np.float64).reshape(1, -1)
        results = []
        key = "cap_feature" if use_caption else "feature"
        for node, attrs in self.nodes(data=True):
            feat = attrs.get(key)
            if feat is None:
                continue
            feat = np.asarray(feat, dtype=np.float64).reshape(1, -1)
            feat = feat / (np.linalg.norm(feat) + _EPS)
            sim = float((feat @ query_feat.T).item() / (np.linalg.norm(query_feat) + _EPS))
            results.append((sim, node))
        results.sort(key=lambda x: -x[0])
        return results[:max(int(top_k), 1)]

    def query_by_text(self, text_feature, top_k=5, use_caption=False):
        """Alias of :meth:`query_by_feature` (the caller encodes the text)."""
        return self.query_by_feature(text_feature, top_k=top_k, use_caption=use_caption)

    # ------------------------------------------------------------------ #
    # downstream: occupancy query
    # ------------------------------------------------------------------ #
    def occupancy_query(self, xyz):
        """Return the node id occupying each query point, or None if free."""
        xyz = np.asarray(xyz, dtype=np.float64)
        single = xyz.ndim == 1
        if single:
            xyz = xyz[None, :]
        out = [None] * len(xyz)
        for node, attrs in self.nodes(data=True):
            occ = attrs["octree"].query_occupancy_batch(xyz)
            for i in np.nonzero(occ)[0]:
                if out[i] is None:  # keep the first (coarsest/first-built) instance
                    out[i] = node
        return out[0] if single else out

    # ------------------------------------------------------------------ #
    # downstream: path planning
    # ------------------------------------------------------------------ #
    def plan_path(self, start, goal, resolution=None, margin=0.0,
                  max_waypoints=100000):
        """Plan a collision-free path with A* on the scene's free voxels.

        Args:
            start / goal: (3,) world coordinates.
            resolution: voxel size (defaults to the octree min cell size).
            margin: safety margin (meters) around occupied voxels.
            max_waypoints: bound on expanded nodes.

        Returns:
            (waypoints, info) where waypoints is an (K, 3) array of voxel
            centers from start to goal, or None if no path exists.
        """
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        if resolution is None:
            resolution = self.graph.get("min_cell_size", 0.05)
        resolution = float(resolution)

        # -------------------------------------------------------------- #
        # 1. build the occupancy grid
        # -------------------------------------------------------------- #
        roots = np.stack([a["octree"].root for _, a in self.nodes(data=True)
                          if a["octree"].root is not None]) if len(self) else np.empty((0, 4))
        pts = np.vstack([start, goal])
        if len(roots):
            lo = np.minimum((roots[:, :3] - roots[:, 3, None]).min(axis=0), pts.min(0))
            hi = np.maximum((roots[:, :3] + roots[:, 3, None]).max(axis=0), pts.max(0))
        else:
            lo, hi = pts.min(0), pts.max(0)
        lo -= margin + resolution
        hi += margin + resolution

        shape = np.ceil((hi - lo) / resolution).astype(np.int64) + 1
        shape = tuple(np.clip(shape, 2, 512).tolist())  # bound the grid size
        grid = np.zeros(shape, dtype=np.int8)

        xs = lo[0] + (np.arange(shape[0]) + 0.5) * resolution
        ys = lo[1] + (np.arange(shape[1]) + 0.5) * resolution
        zs = lo[2] + (np.arange(shape[2]) + 0.5) * resolution
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        voxel_centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        for _, attrs in self.nodes(data=True):
            attrs["octree"].mark_grid(voxel_centers, shape, grid)
        if margin > 0:
            grid = _inflate_grid(grid, max(int(np.ceil(margin / resolution)), 1))

        # -------------------------------------------------------------- #
        # 2. snap start / goal to the nearest free voxel
        # -------------------------------------------------------------- #
        def to_voxel(p):
            v = np.floor((p - lo) / resolution).astype(np.int64)
            return tuple(np.clip(v, 0, np.array(shape) - 1))

        def nearest_free(p):
            v = to_voxel(p)
            if not grid[v]:
                return v
            # BFS in voxel space
            seen = {v}
            queue = [v]
            while queue:
                cur = queue.pop(0)
                for nxt in _neighbors(cur, shape):
                    if nxt in seen:
                        continue
                    if not grid[nxt]:
                        return nxt
                    seen.add(nxt)
                    queue.append(nxt)
            return None

        s = nearest_free(start)
        g = nearest_free(goal)
        if s is None or g is None:
            return None, {"reason": "start or goal is trapped in occupied space"}

        # -------------------------------------------------------------- #
        # 3. A* search (26-connected)
        # -------------------------------------------------------------- #
        def heuristic(a, b):
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))

        open_set = [(heuristic(s, g), 0, s)]
        came = {s: None}
        gscore = {s: 0}
        expanded = 0
        while open_set and expanded < max_waypoints:
            _, cur_g, cur = heapq.heappop(open_set)
            if cur == g:
                # reconstruct path
                path = [cur]
                while came[cur] is not None:
                    cur = came[cur]
                    path.append(cur)
                path = path[::-1]
                waypoints = lo + (np.array(path, dtype=np.float64) + 0.5) * resolution
                # replace the endpoints with the true start / goal
                waypoints[0] = start
                waypoints[-1] = goal
                return waypoints, {
                    "grid_shape": list(shape),
                    "resolution": resolution,
                    "expanded_nodes": expanded,
                }
            expanded += 1
            for nxt in _neighbors(cur, shape):
                if grid[nxt]:
                    continue
                ng = cur_g + 1
                if ng < gscore.get(nxt, np.inf):
                    gscore[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(open_set, (ng + heuristic(nxt, g), ng, nxt))
        return None, {"reason": "no path found within budget"}

    # ------------------------------------------------------------------ #
    # io & stats
    # ------------------------------------------------------------------ #
    def state_dict(self):
        d = {
            "graph": dict(self.graph),
            "nodes": [],
            "edges": [],
        }
        for node, attrs in self.nodes(data=True):
            nd = dict(attrs)
            nd["node"] = node
            nd["octree"] = attrs["octree"].state_dict()
            d["nodes"].append(nd)
        for u, v, attrs in self.edges(data=True):
            d["edges"].append((u, v, attrs))
        return d

    @classmethod
    def from_state_dict(cls, d):
        G = cls()
        G.graph.update(d["graph"])
        for nd in d["nodes"]:
            node = nd.pop("node")
            octree = AdaptiveOctree.from_state_dict(nd.pop("octree"))
            G.add_node(node, **nd, octree=octree)
        for u, v, attrs in d["edges"]:
            G.add_edge(u, v, **attrs)
        return G

    def save(self, path):
        import pickle
        with open(path, "wb") as fp:
            pickle.dump(self.state_dict(), fp)

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, "rb") as fp:
            return cls.from_state_dict(pickle.load(fp))

    def storage_bytes(self):
        total = 0
        for _, attrs in self.nodes(data=True):
            total += attrs["octree"].bytes_estimate()
        return total


def _neighbors(voxel, shape):
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                n = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                if all(0 <= n[i] < shape[i] for i in range(3)):
                    yield n


def _inflate_grid(grid, radius):
    """Dilate occupied voxels by ``radius`` (Chebyshev distance)."""
    from scipy.ndimage import maximum_filter
    return maximum_filter(grid, size=2 * radius + 1, mode="constant", cval=0).astype(np.int8)
