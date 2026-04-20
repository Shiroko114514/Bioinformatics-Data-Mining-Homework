import math
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple, Union

from splice_utils import (
    _marginals,
    learn_cpts,
    compute_mi_matrix,
    chow_liu_tree,
    validate_seqs,
    BASES,
    BASE_IDX,
)

class BayesianNetworkModel:
    def __init__(
        self,
        window: int,
        site: str = 'donor',
        structure: Union[str, List[int]] = 'chow-liu',
        dependency_threshold: float = 6.0,
        max_parents: int = 2,
    ) -> None:
        self.window = window
        self.site = site
        self.structure = structure
        self.dependency_threshold = float(dependency_threshold)
        self.max_parents = max(1, int(max_parents))

        # Backward-compatible single-parent representation (used by chain/chow-liu/list).
        self.parents: Optional[List[int]] = None

        # General multi-parent representation (used by EBN and projected from single-parent structures).
        self.parent_sets: Optional[List[List[int]]] = None

        # Parameters for single-parent mode.
        self.root_fg: Optional[Dict[str, float]] = None
        self.cpt_fg: Optional[List] = None
        self.root_bg: Optional[Dict[str, float]] = None
        self.cpt_bg: Optional[List] = None

        # Parameters for multi-parent mode.
        self.node_cpt_fg: Optional[List[Dict[Tuple[str, ...], Dict[str, float]]]] = None
        self.node_cpt_bg: Optional[List[Dict[Tuple[str, ...], Dict[str, float]]]] = None

        self.mi_matrix: Optional[List[List[float]]] = None
        self.chi2_matrix: Optional[List[List[float]]] = None
        self._marginals_fg: Optional[List[Dict]] = None

    def _compute_chi2_matrix(self, seqs: List[str], n: int) -> List[List[float]]:
        chi2 = [[0.0] * n for _ in range(n)]
        total = len(seqs)
        if total == 0:
            return chi2

        for i in range(n):
            for j in range(i + 1, n):
                table = [[0.0] * 4 for _ in range(4)]
                for s in seqs:
                    ai = BASE_IDX.get(s[i], -1)
                    bj = BASE_IDX.get(s[j], -1)
                    if ai >= 0 and bj >= 0:
                        table[ai][bj] += 1.0

                row_sum = [sum(table[r]) for r in range(4)]
                col_sum = [sum(table[r][c] for r in range(4)) for c in range(4)]

                score = 0.0
                for r in range(4):
                    for c in range(4):
                        expected = (row_sum[r] * col_sum[c]) / max(float(total), 1.0)
                        if expected > 1e-12:
                            diff = table[r][c] - expected
                            score += (diff * diff) / expected
                chi2[i][j] = score
                chi2[j][i] = score
        return chi2

    def _build_ebn_parent_sets(self, chi2: List[List[float]], n: int) -> List[List[int]]:
        dep_strength = [sum(chi2[i]) for i in range(n)]
        root = max(range(n), key=lambda i: dep_strength[i]) if n > 0 else 0

        neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if chi2[i][j] >= self.dependency_threshold:
                    neighbors[i].append(j)
                    neighbors[j].append(i)

        # Keep graph connected by linking isolated nodes to their strongest partner.
        for i in range(n):
            if neighbors[i]:
                continue
            best_j = max((j for j in range(n) if j != i), key=lambda j: chi2[i][j], default=None)
            if best_j is not None:
                neighbors[i].append(best_j)
                neighbors[best_j].append(i)

        order: List[int] = []
        seen = set()
        q: deque[int] = deque([root])
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            order.append(u)
            nxt = sorted(neighbors[u], key=lambda v: chi2[u][v], reverse=True)
            for v in nxt:
                if v not in seen:
                    q.append(v)

        if len(order) < n:
            rest = [i for i in range(n) if i not in seen]
            rest.sort(key=lambda i: dep_strength[i], reverse=True)
            order.extend(rest)

        parent_sets: List[List[int]] = [[] for _ in range(n)]
        for idx in range(1, len(order)):
            node = order[idx]
            prev = order[:idx]
            prev.sort(key=lambda p: chi2[node][p], reverse=True)
            parent_sets[node] = prev[: self.max_parents]
        return parent_sets

    def _learn_multi_parent_cpts(
        self,
        seqs: List[str],
        parent_sets: List[List[int]],
        n: int,
    ) -> List[Dict[Tuple[str, ...], Dict[str, float]]]:
        cpts: List[Dict[Tuple[str, ...], Dict[str, float]]] = [dict() for _ in range(n)]
        for i in range(n):
            parents = parent_sets[i]
            grouped: Dict[Tuple[str, ...], Counter[str]] = {}
            for seq in seqs:
                key = tuple(seq[p] for p in parents)
                grouped.setdefault(key, Counter())
                grouped[key][seq[i]] += 1

            if not grouped:
                grouped[tuple()] = Counter()

            node_cpt: Dict[Tuple[str, ...], Dict[str, float]] = {}
            for key, cnt in grouped.items():
                total = float(sum(cnt.values())) + 4.0
                node_cpt[key] = {b: (float(cnt.get(b, 0.0)) + 1.0) / total for b in BASES}
            cpts[i] = node_cpt
        return cpts

    def _project_single_parent_sets(self, parents: List[int]) -> List[List[int]]:
        return [[] if p == -1 else [p] for p in parents]

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        pos = validate_seqs(pos_seqs, self.window)
        neg = validate_seqs(neg_seqs, self.window)
        if not pos:
            raise ValueError("No valid positive sequences after filtering.")

        n = self.window

        if isinstance(self.structure, list):
            self.parents = list(self.structure)
            self.parent_sets = self._project_single_parent_sets(self.parents)
        elif self.structure == 'chain':
            self.parents = [-1] + list(range(n - 1))
            self.parent_sets = self._project_single_parent_sets(self.parents)
        elif self.structure == 'chow-liu':
            self.mi_matrix = compute_mi_matrix(pos, n)
            self.parents = chow_liu_tree(self.mi_matrix, n, root=0)
            self.parent_sets = self._project_single_parent_sets(self.parents)
        elif self.structure == 'ebn':
            self.chi2_matrix = self._compute_chi2_matrix(pos, n)
            self.parent_sets = self._build_ebn_parent_sets(self.chi2_matrix, n)
            # Keep a representative single parent for compatibility with helper methods.
            self.parents = [ps[0] if ps else -1 for ps in self.parent_sets]
        elif self.structure == 'greedy-bic':
            raise NotImplementedError('greedy-bic is not implemented in this task; use chow-liu or ebn.')
        else:
            raise ValueError(f"Unknown structure: {self.structure!r}")

        self._marginals_fg = _marginals(pos, n)
        bg_seqs = neg if neg else pos

        if self.structure == 'ebn':
            if self.parent_sets is None:
                raise RuntimeError('Internal error: parent sets not built for EBN.')
            self.node_cpt_fg = self._learn_multi_parent_cpts(pos, self.parent_sets, n)
            self.node_cpt_bg = self._learn_multi_parent_cpts(bg_seqs, self.parent_sets, n)
            self.root_fg, self.cpt_fg = None, None
            self.root_bg, self.cpt_bg = None, None
        else:
            if self.parents is None:
                raise RuntimeError('Internal error: parents not built for single-parent BN.')
            self.root_fg, self.cpt_fg = learn_cpts(pos, self.parents, n)
            self.root_bg, self.cpt_bg = learn_cpts(bg_seqs, self.parents, n)
            self.node_cpt_fg, self.node_cpt_bg = None, None

    def _log_prob(self, seq: str, root_marg: Dict,
                  cpt: List) -> float:
        lp = 0.0
        for i in range(self.window):
            b = seq[i]
            if b not in BASE_IDX:
                return float('-inf')
            pa = self.parents[i]
            if pa == -1:
                prob = root_marg.get(b, 1e-10)
            else:
                pb = seq[pa]
                prob = cpt[i][pb].get(b, 1e-10) if pb in BASE_IDX else 1e-10
            lp += math.log(max(prob, 1e-10))
        return lp

    def _log_prob_multi(self, seq: str, node_cpt: List[Dict[Tuple[str, ...], Dict[str, float]]]) -> float:
        if self.parent_sets is None:
            raise RuntimeError('Model not trained with parent sets.')
        lp = 0.0
        for i in range(self.window):
            b = seq[i]
            if b not in BASE_IDX:
                return float('-inf')
            parents = self.parent_sets[i]
            key = tuple(seq[p] for p in parents)
            cond = node_cpt[i].get(key)
            p = cond.get(b, 0.25) if cond is not None else 0.25
            lp += math.log(max(p, 1e-10))
        return lp

    def score(self, seq: str) -> float:
        if self.parents is None and self.parent_sets is None:
            raise RuntimeError("Model not trained. Call train() first.")
        seq = seq.upper()
        if len(seq) != self.window:
            raise ValueError(f"Expected length {self.window}, got {len(seq)}")
        if self.structure == 'ebn':
            if self.node_cpt_fg is None or self.node_cpt_bg is None:
                raise RuntimeError('EBN parameters are missing; call train() first.')
            return self._log_prob_multi(seq, self.node_cpt_fg) - self._log_prob_multi(seq, self.node_cpt_bg)
        if self.root_fg is None or self.cpt_fg is None or self.root_bg is None or self.cpt_bg is None:
            raise RuntimeError('BN parameters are missing; call train() first.')
        return self._log_prob(seq, self.root_fg, self.cpt_fg) - self._log_prob(seq, self.root_bg, self.cpt_bg)

    def score_batch(self, seqs: List[str]) -> List[float]:
        return [self.score(s) for s in seqs]

    def non_adjacent_edges(self) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        if self.parent_sets is not None:
            for child, ps in enumerate(self.parent_sets):
                for parent in ps:
                    if abs(child - parent) > 1:
                        edges.append((child, parent))
            return edges
        if self.parents is None:
            return edges
        for child, parent in enumerate(self.parents):
            if parent >= 0 and abs(child - parent) > 1:
                edges.append((child, parent))
        return edges

    def print_summary(self) -> None:
        print("\nBayesian Network Splice Model")
        print("=" * 48)
        print(f"  Site          : {self.site}")
        print(f"  Window        : {self.window}")
        print(f"  Structure     : {self.structure}")
        if self.structure == 'ebn':
            print(f"  Max parents p : {self.max_parents}")
            print(f"  Chi2 threshold: {self.dependency_threshold:.2f}")
        edge_count = 0
        if self.parent_sets is not None:
            edge_count = sum(len(ps) for ps in self.parent_sets)
        elif self.parents is not None:
            edge_count = sum(1 for p in self.parents if p != -1)
        print(f"  Edge count    : {edge_count}")
        print(f"  Long-range    : {len(self.non_adjacent_edges())}")
        print("=" * 48)

class BNScanner:
    DONOR_OFFSET = 3
    ACCEPTOR_OFFSET = 20

    def __init__(self, model: BayesianNetworkModel, threshold: float = 0.0) -> None:
        self.model = model
        self.threshold = threshold
        self.site = model.site
        self.offset = self.DONOR_OFFSET if self.site == 'donor' else self.ACCEPTOR_OFFSET
        self.dinuc = 'GT' if self.site == 'donor' else 'AG'

    def scan(self, genome: str):
        genome = genome.upper()
        w = self.model.window
        off = self.offset
        hits = []
        for i in range(off, len(genome) - w + off + 1):
            win = genome[i - off: i - off + w]
            if len(win) != w:
                continue
            if win[off:off + 2] != self.dinuc:
                continue
            sc = self.model.score(win)
            if sc >= self.threshold:
                hits.append((i, sc))
        return hits

class _WMMModel:
    def __init__(self, window):
        self.window = window
        self.log_odds = None

    def train(self, pos_seqs, neg_seqs):
        from .splice_utils import empirical_bg, BASES, PSEUDOCOUNT
        pos = validate_seqs(pos_seqs, self.window)
        n = self.window
        bg = empirical_bg(neg_seqs) if neg_seqs else {b: 0.25 for b in BASES}
        cts = [{b: PSEUDOCOUNT for b in BASES} for _ in range(n)]
        for seq in pos:
            for i, c in enumerate(seq):
                cts[i][c] += 1
        self.log_odds = []
        for pos_ct in cts:
            total = sum(pos_ct.values())
            freq = {b: pos_ct[b] / total for b in BASES}
            self.log_odds.append({b: math.log(max(freq[b], 1e-10) / max(bg[b], 1e-10)) for b in BASES})

    def score(self, seq):
        seq = seq.upper()
        return sum(self.log_odds[i].get(c, 0.0) for i, c in enumerate(seq) if c in BASE_IDX)

    def score_batch(self, seqs):
        return [self.score(s) for s in seqs]

class _WAMModel:
    def __init__(self, window):
        self.window = window
        self._bn = BayesianNetworkModel(window, structure='chain')

    def train(self, pos_seqs, neg_seqs):
        self._bn.train(pos_seqs, neg_seqs)

    def score(self, seq):
        return self._bn.score(seq)

    def score_batch(self, seqs):
        return self._bn.score_batch(seqs)
