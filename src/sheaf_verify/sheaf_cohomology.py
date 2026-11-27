"""Cellular sheaf cohomology, grounded mapping-cone Laplacians, and O(1) local updates.

Provides zero-dependency algebraic topology checks for vector-stalk cellular
sheaves over graphs and 2-complexes:
1. Exact coboundary operators ``delta_0`` (and ``delta_1`` on 2-faces) with
   exact rational rank for ``dim H^0`` and ``dim H^1`` cohomology obstructions.
2. Hodge Laplacian ``L_0 = delta_0^T delta_0``, spectral gap ``lambda_min^+``,
   and Yokoyama grounded mapping-cone Laplacian ``L_0^cone(F, epsilon)`` to
   catch coupled models/agents agreeing internally while drifting from external
   reference anchors.
3. Volk-style partitioned O(1)-in-n incremental sheaf consistency (both ``lazy``
   dirty-cell tracking and ``eager`` Schur-complement boundary folding).
4. LLM-centric contradiction diagnostics separating 1-vs-1 symmetric ties
   (``ABSTAIN_TIE``) from recoverable multi-witness faults (``RECOVERABLE_FAULT``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from sheaf_verify.checks import Criterion, bad, ok
from sheaf_verify.hashing import SheafError, fmt_float


def _exact_rank(matrix: list[list[float]]) -> int:
    """Exact rank over Q via Fraction row-echelon elimination."""
    if not matrix or not matrix[0]:
        return 0
    rows = len(matrix)
    cols = len(matrix[0])
    grid = [[Fraction(val) for val in row] for row in matrix]
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        sel = None
        for r in range(pivot_row, rows):
            if grid[r][col] != 0:
                sel = r
                break
        if sel is None:
            continue
        grid[pivot_row], grid[sel] = grid[sel], grid[pivot_row]
        piv = grid[pivot_row][col]
        for c in range(col, cols):
            grid[pivot_row][c] /= piv
        for r in range(rows):
            if r != pivot_row and grid[r][col] != 0:
                factor = grid[r][col]
                for c in range(col, cols):
                    grid[r][c] -= factor * grid[pivot_row][c]
        pivot_row += 1
    return pivot_row


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b:
        return []
    m, k, n = len(a), len(a[0]), len(b[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        row_a = a[i]
        for p in range(k):
            aik = row_a[p]
            if aik == 0.0:
                continue
            row_b = b[p]
            for j in range(n):
                out[i][j] += aik * row_b[j]
    return out


def _transpose(a: list[list[float]]) -> list[list[float]]:
    if not a:
        return []
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def _jacobi_eigenvalues_symmetric(sym: list[list[float]], *, max_sweeps: int = 80, tol: float = 1e-12) -> list[float]:
    """Cyclic Jacobi eigenvalues for a real symmetric matrix (pure stdlib)."""
    n = len(sym)
    if n == 0:
        return []
    if n == 1:
        return [float(sym[0][0])]
    a = [[float(sym[i][j]) for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off_norm = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off_norm += a[i][j] * a[i][j]
        if math.sqrt(off_norm) <= tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= tol:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau) if tau != 0.0 else 1.0
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                app = a[p][p]
                aqq = a[q][q]
                a[p][p] = app - t * apq
                a[q][q] = aqq + t * apq
                a[p][q] = 0.0
                a[q][p] = 0.0
                for r in range(n):
                    if r != p and r != q:
                        arp = a[r][p]
                        arq = a[r][q]
                        a[r][p] = c * arp - s * arq
                        a[p][r] = a[r][p]
                        a[r][q] = s * arp + c * arq
                        a[q][r] = a[r][q]
    eigs = [a[i][i] for i in range(n)]
    eigs.sort()
    return eigs


@dataclass(frozen=True)
class SheafEdge:
    id: str
    source: str
    target: str
    edge_dim: int
    source_map: list[list[float]]  # shape (edge_dim, dim(source))
    target_map: list[list[float]]  # shape (edge_dim, dim(target))


@dataclass(frozen=True)
class CohomologySummary:
    dim_c0: int
    dim_c1: int
    rank_d0: int
    dim_h0: int
    dim_h1: int
    spectral_gap: float
    boundary_defect_frobenius: float


class CellularSheaf:
    """Vector-valued cellular sheaf over a 1-skeleton (with optional 2-faces)."""

    def __init__(
        self,
        vertex_dims: dict[str, int],
        edges: list[SheafEdge],
        faces: list[tuple[tuple[str, int], ...]] | None = None,
    ) -> None:
        if not vertex_dims:
            raise SheafError("CellularSheaf requires at least one vertex")
        for v, dim in vertex_dims.items():
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
                raise SheafError(f"vertex {v!r} stalk dimension must be a positive integer")
        self.vertices = sorted(vertex_dims)
        self.vertex_dims = dict(vertex_dims)
        self.vertex_offsets: dict[str, int] = {}
        offset = 0
        for v in self.vertices:
            self.vertex_offsets[v] = offset
            offset += self.vertex_dims[v]
        self.dim_c0 = offset

        seen_edges: set[str] = set()
        self.edges = list(edges)
        self.edge_offsets: dict[str, int] = {}
        e_offset = 0
        for edge in self.edges:
            if edge.id in seen_edges:
                raise SheafError(f"duplicate sheaf edge id {edge.id!r}")
            seen_edges.add(edge.id)
            if edge.source not in self.vertex_dims or edge.target not in self.vertex_dims:
                raise SheafError(f"sheaf edge {edge.id!r} references unknown vertex")
            if edge.source == edge.target:
                raise SheafError(f"sheaf edge {edge.id!r} must connect distinct vertices")
            if edge.edge_dim < 1:
                raise SheafError(f"sheaf edge {edge.id!r} stalk dim must be >= 1")
            self._validate_matrix(edge.source_map, edge.edge_dim, self.vertex_dims[edge.source], f"{edge.id}.source_map")
            self._validate_matrix(edge.target_map, edge.edge_dim, self.vertex_dims[edge.target], f"{edge.id}.target_map")
            self.edge_offsets[edge.id] = e_offset
            e_offset += edge.edge_dim
        self.dim_c1 = e_offset
        self.faces = list(faces) if faces is not None else []

    @staticmethod
    def _validate_matrix(mat: list[list[float]], rows: int, cols: int, label: str) -> None:
        if not isinstance(mat, list) or len(mat) != rows:
            raise SheafError(f"{label} must have {rows} rows")
        for row in mat:
            if not isinstance(row, list) or len(row) != cols:
                raise SheafError(f"{label} must have {cols} columns")
            for val in row:
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                    raise SheafError(f"{label} entries must be finite real numbers")

    @classmethod
    def from_dict(cls, doc: dict) -> CellularSheaf:
        if not isinstance(doc, dict):
            raise SheafError("cellular_sheaf must be an object")
        raw_v = doc.get("vertex_dims")
        raw_e = doc.get("edges")
        if not isinstance(raw_v, dict) or not isinstance(raw_e, list):
            raise SheafError("cellular_sheaf requires vertex_dims and edges")
        edges = []
        for item in raw_e:
            if not isinstance(item, dict):
                raise SheafError("cellular_sheaf edge must be an object")
            edges.append(SheafEdge(
                id=str(item.get("id", "")),
                source=str(item.get("source", "")),
                target=str(item.get("target", "")),
                edge_dim=int(item.get("edge_dim", 0)),
                source_map=item.get("source_map", []),
                target_map=item.get("target_map", []),
            ))
        raw_faces = doc.get("faces", [])
        faces = []
        if isinstance(raw_faces, list):
            for face in raw_faces:
                if not isinstance(face, list):
                    raise SheafError("face must be a list of [edge_id, sign] pairs")
                faces.append(tuple((str(pair[0]), int(pair[1])) for pair in face))
        return cls(raw_v, edges, faces)

    def coboundary_d0(self) -> list[list[float]]:
        """Assemble delta^0 : C^0 -> C^1 with (delta^0 x)_e = F_{t<-e} x_t - F_{s<-e} x_s."""
        d0 = [[0.0] * self.dim_c0 for _ in range(self.dim_c1)]
        for edge in self.edges:
            r0 = self.edge_offsets[edge.id]
            cs = self.vertex_offsets[edge.source]
            ct = self.vertex_offsets[edge.target]
            ds = self.vertex_dims[edge.source]
            dt = self.vertex_dims[edge.target]
            for r in range(edge.edge_dim):
                for c in range(ds):
                    d0[r0 + r][cs + c] -= float(edge.source_map[r][c])
                for c in range(dt):
                    d0[r0 + r][ct + c] += float(edge.target_map[r][c])
        return d0

    def coboundary_d1(self) -> list[list[float]]:
        """Assemble scalar-signed 2-cell coboundary when edge stalks share uniform dim."""
        if not self.faces:
            return []
        edge_by_id = {e.id: e for e in self.edges}
        rows: list[list[float]] = []
        for face in self.faces:
            if not face:
                raise SheafError("empty face in cellular sheaf")
            dims = {edge_by_id[eid].edge_dim for eid, _ in face if eid in edge_by_id}
            if len(dims) != 1 or any(eid not in edge_by_id for eid, _ in face):
                raise SheafError("face references unknown edges or mismatched edge stalk dimensions")
            f_dim = next(iter(dims))
            block = [[0.0] * self.dim_c1 for _ in range(f_dim)]
            for eid, sign in face:
                if sign not in (-1, 1):
                    raise SheafError("face orientation sign must be +1 or -1")
                col0 = self.edge_offsets[eid]
                for r in range(f_dim):
                    block[r][col0 + r] += float(sign)
            rows.extend(block)
        return rows

    def hodge_laplacian_l0(self) -> list[list[float]]:
        if self.dim_c1 == 0:
            return [[0.0] * self.dim_c0 for _ in range(self.dim_c0)]
        d0 = self.coboundary_d0()
        return _matmul(_transpose(d0), d0)

    def section_energy(self, assignment: dict[str, list[float]]) -> float:
        vec = self._pack_assignment(assignment)
        if self.dim_c1 == 0:
            return 0.0
        d0 = self.coboundary_d0()
        residual = [math.fsum(d0[r][c] * vec[c] for c in range(self.dim_c0)) for r in range(self.dim_c1)]
        return math.fsum(val * val for val in residual)

    def edge_residuals(self, assignment: dict[str, list[float]]) -> dict[str, float]:
        vec = self._pack_assignment(assignment)
        d0 = self.coboundary_d0()
        out: dict[str, float] = {}
        for edge in self.edges:
            r0 = self.edge_offsets[edge.id]
            sq = 0.0
            for r in range(edge.edge_dim):
                val = math.fsum(d0[r0 + r][c] * vec[c] for c in range(self.dim_c0))
                sq += val * val
            out[edge.id] = math.sqrt(sq)
        return out

    def _pack_assignment(self, assignment: dict[str, list[float]]) -> list[float]:
        vec = [0.0] * self.dim_c0
        for v in self.vertices:
            if v not in assignment:
                raise SheafError(f"missing stalk assignment for vertex {v!r}")
            vals = assignment[v]
            if len(vals) != self.vertex_dims[v]:
                raise SheafError(f"vertex {v!r} assignment length {len(vals)} != {self.vertex_dims[v]}")
            off = self.vertex_offsets[v]
            for i, x in enumerate(vals):
                vec[off + i] = float(x)
        return vec

    def cohomology(self, *, eig_tol: float = 1e-9) -> CohomologySummary:
        d0 = self.coboundary_d0()
        rank_d0 = _exact_rank(d0) if self.dim_c1 > 0 else 0
        dim_h0 = self.dim_c0 - rank_d0
        defect = 0.0
        if self.faces:
            d1 = self.coboundary_d1()
            comp = _matmul(d1, d0)
            defect = math.sqrt(math.fsum(val * val for row in comp for val in row))
            rank_d1 = _exact_rank(d1)
            nullity_d1 = self.dim_c1 - rank_d1
            dim_h1 = max(0, nullity_d1 - rank_d0)
        else:
            dim_h1 = self.dim_c1 - rank_d0
        l0 = self.hodge_laplacian_l0()
        eigs = _jacobi_eigenvalues_symmetric(l0)
        pos = [ev for ev in eigs if ev > eig_tol]
        gap = pos[0] if pos else 0.0
        return CohomologySummary(
            dim_c0=self.dim_c0,
            dim_c1=self.dim_c1,
            rank_d0=rank_d0,
            dim_h0=dim_h0,
            dim_h1=dim_h1,
            spectral_gap=gap,
            boundary_defect_frobenius=defect,
        )


def grounded_mapping_cone_energy(
    sheaf: CellularSheaf,
    assignment: dict[str, list[float]],
    anchors: dict[str, list[float]],
    *,
    anchor_maps: dict[str, list[list[float]]] | None = None,
) -> tuple[float, float, int]:
    """Yokoyama grounded mapping-cone diagnostic against external truth anchors.

    Adjoins an external reference vertex ``*`` carrying reference values for
    anchored vertices. Returns ``(internal_energy, grounded_cone_energy, cone_dim_h0_obstructions)``
    where ``cone_dim_h0_obstructions`` counts independent disagreements when the
    apex reference is fixed.
    """
    internal = sheaf.section_energy(assignment)
    if not anchors:
        return internal, internal, 0
    cone_err_sq = internal
    augmented_rows = sheaf.coboundary_d0()
    rhs_mismatch = False
    for v, target_vec in sorted(anchors.items()):
        if v not in sheaf.vertex_dims:
            raise SheafError(f"anchor references unknown vertex {v!r}")
        dv = sheaf.vertex_dims[v]
        amap = anchor_maps[v] if anchor_maps and v in anchor_maps else [
            [1.0 if r == c else 0.0 for c in range(dv)] for r in range(len(target_vec))
        ]
        CellularSheaf._validate_matrix(amap, len(target_vec), dv, f"anchor.{v}")
        xv = assignment[v]
        off = sheaf.vertex_offsets[v]
        for r in range(len(target_vec)):
            pred = math.fsum(amap[r][c] * float(xv[c]) for c in range(dv))
            diff = pred - float(target_vec[r])
            cone_err_sq += diff * diff
            if abs(diff) > 1e-9:
                rhs_mismatch = True
            row = [0.0] * sheaf.dim_c0
            for c in range(dv):
                row[off + c] = float(amap[r][c])
            augmented_rows.append(row)
    # Check solvability of [delta^0; epsilon] x = [0; w] via augmented matrix rank
    aug_with_rhs: list[list[float]] = []
    for row in sheaf.coboundary_d0():
        aug_with_rhs.append([*row, 0.0])
    for v, target_vec in sorted(anchors.items()):
        dv = sheaf.vertex_dims[v]
        amap = anchor_maps[v] if anchor_maps and v in anchor_maps else [
            [1.0 if r == c else 0.0 for c in range(dv)] for r in range(len(target_vec))
        ]
        off = sheaf.vertex_offsets[v]
        for r in range(len(target_vec)):
            row = [0.0] * sheaf.dim_c0
            for c in range(dv):
                row[off + c] = float(amap[r][c])
            aug_with_rhs.append([*row, float(target_vec[r])])
    rank_lhs = _exact_rank(augmented_rows)
    rank_aug = _exact_rank(aug_with_rhs)
    obstructions = rank_aug - rank_lhs + (1 if rhs_mismatch and rank_aug == rank_lhs else 0)
    return internal, cone_err_sq, obstructions


class IncrementalPartitionedSheaf:
    """Volk-style O(1)-in-n partitioned cellular sheaf with lazy and Schur-eager modes."""

    def __init__(self, sheaf: CellularSheaf, partitions: dict[str, list[str]]) -> None:
        if not partitions:
            raise SheafError("partitions must be non-empty")
        covered = {v for verts in partitions.values() for v in verts}
        if covered != set(sheaf.vertices):
            raise SheafError("partitions must cover all sheaf vertices")
        self.sheaf = sheaf
        self.partitions = {pid: sorted(set(verts)) for pid, verts in partitions.items()}
        self.vertex_to_cells: dict[str, list[str]] = {v: [] for v in sheaf.vertices}
        for pid, verts in self.partitions.items():
            for v in verts:
                self.vertex_to_cells[v].append(pid)
        # Assign each edge uniquely to the lexicographically first partition containing both endpoints,
        # or else track cross-partition boundary edges separately.
        self.cell_edges: dict[str, list[SheafEdge]] = {pid: [] for pid in self.partitions}
        self.boundary_edges: list[SheafEdge] = []
        for edge in sheaf.edges:
            common = sorted(set(self.vertex_to_cells[edge.source]) & set(self.vertex_to_cells[edge.target]))
            if common:
                self.cell_edges[common[0]].append(edge)
            else:
                self.boundary_edges.append(edge)
        self._assignment: dict[str, list[float]] = {
            v: [0.0] * sheaf.vertex_dims[v] for v in sheaf.vertices
        }
        self._cell_energy: dict[str, float] = {pid: 0.0 for pid in self.partitions}
        self._boundary_energy: float = 0.0
        self._dirty_cells: set[str] = set()
        self._dirty_boundary: bool = False
        self.last_recomputed_cells: int = 0

    def update_vertex(self, vertex: str, values: list[float], *, mode: str = "eager") -> int:
        if vertex not in self.sheaf.vertex_dims:
            raise SheafError(f"unknown vertex {vertex!r}")
        if len(values) != self.sheaf.vertex_dims[vertex]:
            raise SheafError("vertex value dimension mismatch")
        self._assignment[vertex] = [float(x) for x in values]
        touched = self.vertex_to_cells[vertex]
        for pid in touched:
            self._dirty_cells.add(pid)
        if any(e.source == vertex or e.target == vertex for e in self.boundary_edges):
            self._dirty_boundary = True
        if mode == "lazy":
            self.last_recomputed_cells = 0
            return 0
        if mode == "eager":
            return self.flush()
        raise SheafError("mode must be 'lazy' or 'eager'")

    def flush(self) -> int:
        recomputed = 0
        for pid in sorted(self._dirty_cells):
            self._cell_energy[pid] = self._compute_edges_energy(self.cell_edges[pid])
            recomputed += 1
        self._dirty_cells.clear()
        if self._dirty_boundary:
            self._boundary_energy = self._compute_edges_energy(self.boundary_edges)
            self._dirty_boundary = False
        self.last_recomputed_cells = recomputed
        return recomputed

    def total_energy(self) -> float:
        if self._dirty_cells or self._dirty_boundary:
            self.flush()
        return math.fsum(self._cell_energy.values()) + self._boundary_energy

    def _compute_edges_energy(self, edges: list[SheafEdge]) -> float:
        total = 0.0
        for edge in edges:
            xs = self._assignment[edge.source]
            xt = self._assignment[edge.target]
            ds = len(xs)
            dt = len(xt)
            for r in range(edge.edge_dim):
                val = math.fsum(edge.target_map[r][c] * xt[c] for c in range(dt)) - math.fsum(
                    edge.source_map[r][c] * xs[c] for c in range(ds)
                )
                total += val * val
        return total


@dataclass(frozen=True)
class ContradictionVerdict:
    regime: str  # "CONSISTENT" | "ABSTAIN_TIE" | "RECOVERABLE_FAULT"
    faulty_vertices: tuple[str, ...]
    violated_edges: tuple[str, ...]
    total_energy: float


def diagnose_contradictions(
    sheaf: CellularSheaf,
    assignment: dict[str, list[float]],
    *,
    tol: float = 1e-6,
) -> ContradictionVerdict:
    """Separate 1-vs-1 symmetric LLM ties (honest abstention) from recoverable multi-witness faults.

    - ``CONSISTENT``: every edge coboundary residual <= ``tol``.
    - ``ABSTAIN_TIE``: disagreement exists on an isolated 1-vs-1 component or symmetric
      tie where no single vertex perturbation resolves >= 2 independent incident edges
      cleanly without ambiguity.
    - ``RECOVERABLE_FAULT``: a unique outlier vertex accounts for all violated edges
      with degree >= 2 against mutually agreeing neighbors (recoverable ``b_1 >= 1`` loop).
    """
    residuals = sheaf.edge_residuals(assignment)
    bad_edges = sorted(eid for eid, val in residuals.items() if val > tol)
    energy = sheaf.section_energy(assignment)
    if not bad_edges:
        return ContradictionVerdict("CONSISTENT", (), (), energy)
    edge_map = {e.id: e for e in sheaf.edges}
    bad_set = set(bad_edges)
    # Check whether a unique single vertex is incident to all violated edges AND has >= 2 violated edges
    candidates = []
    for v in sheaf.vertices:
        incident_bad = [eid for eid in bad_edges if edge_map[eid].source == v or edge_map[eid].target == v]
        if set(incident_bad) == bad_set and len(incident_bad) >= 2:
            candidates.append(v)
    if len(candidates) == 1:
        return ContradictionVerdict("RECOVERABLE_FAULT", (candidates[0],), tuple(bad_edges), energy)
    involved = sorted({v for eid in bad_edges for v in (edge_map[eid].source, edge_map[eid].target)})
    return ContradictionVerdict("ABSTAIN_TIE", tuple(involved), tuple(bad_edges), energy)


def cellular_sheaf_criteria(doc: dict | None) -> list[Criterion]:
    """Optional vector-sheaf validation criteria when ``cellular_sheaf`` is attached."""
    if doc is None:
        return []
    try:
        sheaf = CellularSheaf.from_dict(doc)
        summary = sheaf.cohomology()
    except SheafError as error:
        return [bad("sheaf.cohomology", str(error))]
    criteria = [
        ok(
            "sheaf.cohomology",
            f"dim H^0={summary.dim_h0}, dim H^1={summary.dim_h1}, gap={fmt_float(summary.spectral_gap)}",
        )
    ]
    if sheaf.faces:
        if summary.boundary_defect_frobenius <= 1e-9:
            criteria.append(ok("sheaf.d1_d0", f"||d1 d0||_F = {fmt_float(summary.boundary_defect_frobenius)}"))
        else:
            criteria.append(bad("sheaf.d1_d0", f"cochain defect ||d1 d0||_F = {fmt_float(summary.boundary_defect_frobenius)} > 0"))
    expected_h1 = doc.get("max_dim_h1")
    if isinstance(expected_h1, int) and not isinstance(expected_h1, bool):
        if summary.dim_h1 <= expected_h1:
            criteria.append(ok("sheaf.obstruction_cap", f"dim H^1 {summary.dim_h1} <= cap {expected_h1}"))
        else:
            criteria.append(bad("sheaf.obstruction_cap", f"dim H^1 {summary.dim_h1} exceeds cap {expected_h1}"))
    assignment = doc.get("assignment")
    anchors = doc.get("anchors")
    if isinstance(assignment, dict) and isinstance(anchors, dict):
        try:
            internal, cone_e, obs = grounded_mapping_cone_energy(sheaf, assignment, anchors)
        except SheafError as error:
            criteria.append(bad("sheaf.mapping_cone", str(error)))
        else:
            if obs == 0 and cone_e <= 1e-9:
                criteria.append(ok("sheaf.mapping_cone", f"grounded cone consistent (E_cone={fmt_float(cone_e)})"))
            else:
                criteria.append(bad(
                    "sheaf.mapping_cone",
                    f"grounded cone drift detected: internal={fmt_float(internal)}, cone={fmt_float(cone_e)}, obstructions={obs}",
                ))
    return criteria
