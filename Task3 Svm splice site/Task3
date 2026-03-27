"""
Bayesian Network model for eukaryotic splice site prediction
============================================================
References:
  Chow & Liu (1968)   – Approximating discrete probability distributions with
                        dependence trees. IEEE Trans. Inform. Theory 14:462-467.
  Friedman et al.     – Using Bayesian networks to analyze expression data (2000)
  Zhang & Marr (1993) – WAM method (baseline for comparison)

Key ideas
---------
  WMM  : P(s) = ∏ᵢ P(xᵢ)                       — positions independent
  WAM  : P(s) = P(x₀) ∏ᵢ₌₁ P(xᵢ | xᵢ₋₁)        — chain (first-order Markov)
  BN   : P(s) = ∏ᵢ P(xᵢ | Pa(xᵢ))               — DAG, Pa learned from data

Structure learning:
  1. Compute pairwise mutual information MI(i,j) for all position pairs.
  2. Chow-Liu: find maximum spanning tree of the MI graph (Kruskal's MST).
     This yields the optimal tree-structured BN under KL divergence.
  3. (Optional) BIC-based greedy hill-climbing to allow multi-parent nodes.

Scoring:
  log-odds(s) = log P(s | fg BN) − log P(s | bg BN)

Usage
-----
>>> from bayesian_network_splice import BayesianNetworkModel, demo
>>> model = BayesianNetworkModel(window=9, site='donor', structure='chow-liu')
>>> model.train(pos_seqs, neg_seqs)
>>> score = model.score('CAGGTAAGT')
>>> model.print_summary()
>>> demo()
"""

import math
import random
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

BASES      = ('A', 'C', 'G', 'T')
BASE_IDX   = {b: i for i, b in enumerate(BASES)}
PSEUDOCOUNT = 0.5
DONOR_WINDOW    = 9   # exon[-3,-2,-1] | intron[+1..+6]
ACCEPTOR_WINDOW = 23  # intron[-20..-1] | exon[+1..+3]

# ─────────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────────

def validate_seqs(seqs: List[str], window: int) -> List[str]:
    """Filter to valid uppercase ACGT sequences of the correct length."""
    result = []
    for s in seqs:
        s = s.upper().strip()
        if len(s) == window and all(c in BASE_IDX for c in s):
            result.append(s)
    return result

def empirical_bg(seqs: List[str]) -> Dict[str, float]:
    counts: Dict[str, float] = {b: PSEUDOCOUNT for b in BASES}
    total = sum(counts.values())
    for s in seqs:
        for c in s:
            if c in BASE_IDX:
                counts[c] += 1
                total += 1
    return {b: counts[b] / total for b in BASES}

# ─────────────────────────────────────────────────────────────
# Union-Find (for Kruskal's MST)
# ─────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

# ─────────────────────────────────────────────────────────────
# Mutual Information computation
# ─────────────────────────────────────────────────────────────

def _marginals(seqs: List[str], n: int) -> List[Dict[str, float]]:
    """Marginal frequency P(b | position=i) for i in 0..n-1."""
    counts = [{b: PSEUDOCOUNT for b in BASES} for _ in range(n)]
    for seq in seqs:
        for i, c in enumerate(seq[:n]):
            if c in BASE_IDX:
                counts[i][c] += 1
    result = []
    for pos in counts:
        total = sum(pos.values())
        result.append({b: pos[b] / total for b in BASES})
    return result

def _joint(seqs: List[str], i: int, j: int) -> Dict[Tuple[str, str], float]:
    """Joint distribution P(xᵢ=a, xⱼ=b)."""
    counts: Dict[Tuple[str, str], float] = defaultdict(lambda: PSEUDOCOUNT)
    for seq in seqs:
        a, b = seq[i], seq[j]
        if a in BASE_IDX and b in BASE_IDX:
            counts[(a, b)] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def mutual_information(seqs: List[str], i: int, j: int,
                       marginals: List[Dict[str, float]]) -> float:
    """
    MI(Xᵢ, Xⱼ) = Σ_{a,b} P(a,b) log[ P(a,b) / (P(a) P(b)) ]
    Returns MI in nats; always ≥ 0.
    """
    joint = _joint(seqs, i, j)
    mi = 0.0
    for a in BASES:
        for b in BASES:
            pab = joint.get((a, b), 1e-10)
            pa  = marginals[i][a]
            pb  = marginals[j][b]
            if pab > 0 and pa > 0 and pb > 0:
                mi += pab * math.log(pab / (pa * pb))
    return max(mi, 0.0)

def compute_mi_matrix(seqs: List[str], n: int) -> List[List[float]]:
    """Compute the full n×n symmetric MI matrix."""
    marginals = _marginals(seqs, n)
    mi = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m = mutual_information(seqs, i, j, marginals)
            mi[i][j] = m
            mi[j][i] = m
    return mi

# ─────────────────────────────────────────────────────────────
# Chow-Liu: Maximum Spanning Tree → optimal tree-BN
# ─────────────────────────────────────────────────────────────

def chow_liu_tree(mi_matrix: List[List[float]], n: int,
                   root: int = 0) -> List[int]:
    """
    Find the maximum spanning tree of the MI graph (Kruskal's algorithm).
    Orient edges away from `root` (BFS).

    Returns
    -------
    parents : List[int]
        parents[i] = parent of node i; parents[root] = -1.
    """
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((mi_matrix[i][j], i, j))
    edges.sort(reverse=True)          # descending MI

    uf  = _UnionFind(n)
    adj: Dict[int, List[int]] = defaultdict(list)
    for mi_val, u, v in edges:
        if uf.union(u, v):
            adj[u].append(v)
            adj[v].append(u)

    # BFS from root to orient edges
    parents  = [-1] * n
    visited  = [False] * n
    queue    = deque([root])
    visited[root] = True
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                parents[nb] = node
                queue.append(nb)

    return parents

# ─────────────────────────────────────────────────────────────
# BN Parameter Learning (CPTs)
# ─────────────────────────────────────────────────────────────

def learn_cpts(seqs: List[str], parents: List[int], n: int
               ) -> Tuple[Dict[str, float], List[Optional[Dict]]]:
    """
    Estimate conditional probability tables with Laplace smoothing.

    Returns
    -------
    root_marginal : P(b) for the root node
    cpt_list      : cpt_list[i][parent_base][child_base] = P(xᵢ=child | Pa(xᵢ)=parent)
                    cpt_list[root] = None
    """
    root = next(i for i, p in enumerate(parents) if p == -1)

    # Root marginal
    r_counts = {b: PSEUDOCOUNT for b in BASES}
    for seq in seqs:
        b = seq[root]
        if b in BASE_IDX:
            r_counts[b] += 1
    total_r = sum(r_counts.values())
    root_marginal = {b: r_counts[b] / total_r for b in BASES}

    # Conditional tables
    cpt_list: List[Optional[Dict]] = [None] * n
    for i in range(n):
        p = parents[i]
        if p == -1:
            continue
        cond: Dict[str, Dict[str, float]] = {
            pa: {ch: PSEUDOCOUNT for ch in BASES} for pa in BASES
        }
        for seq in seqs:
            pa_b, ch_b = seq[p], seq[i]
            if pa_b in BASE_IDX and ch_b in BASE_IDX:
                cond[pa_b][ch_b] += 1
        cpt: Dict[str, Dict[str, float]] = {}
        for pa_b in BASES:
            total = sum(cond[pa_b].values())
            cpt[pa_b] = {ch_b: cond[pa_b][ch_b] / total for ch_b in BASES}
        cpt_list[i] = cpt

    return root_marginal, cpt_list

# ─────────────────────────────────────────────────────────────
# BIC-based Greedy Structure Search (beyond tree)
# ─────────────────────────────────────────────────────────────

def _has_cycle(parents: List[int], n: int) -> bool:
    """Detect directed cycle via topological sort."""
    in_deg: List[int] = [0] * n
    children: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(parents):
        if p != -1:
            children[p].append(i)
            in_deg[i] += 1
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        node = queue.popleft()
        count += 1
        for ch in children[node]:
            in_deg[ch] -= 1
            if in_deg[ch] == 0:
                queue.append(ch)
    return count != n

def _bic_score(seqs: List[str], parents: List[int], n: int) -> float:
    """
    BIC(G) = log P(D | G, θ̂) − (d_G / 2) × log N

    d_G = Σᵢ (|Xᵢ| − 1) × |PA_i|
        root: |X|−1 = 3  free params
        other: (|X|−1) × |PA| = 3 × 4 = 12 free params
    """
    root_marg, cpt_list = learn_cpts(seqs, parents, n)
    root = next(i for i, p in enumerate(parents) if p == -1)
    N = len(seqs)
    if N == 0:
        return float('-inf')

    # Log-likelihood
    ll = 0.0
    for seq in seqs:
        for i in range(n):
            b = seq[i]
            if b not in BASE_IDX:
                continue
            p = parents[i]
            if p == -1:
                prob = root_marg.get(b, 1e-10)
            else:
                pb = seq[p]
                prob = cpt_list[i][pb].get(b, 1e-10) if pb in BASE_IDX else 1e-10
            ll += math.log(max(prob, 1e-10))

    # Free parameters
    d = len(BASES) - 1  # root
    for i in range(n):
        if parents[i] != -1:
            d += (len(BASES) - 1) * len(BASES)

    return ll - (d / 2.0) * math.log(N)

def greedy_bic_search(seqs: List[str], n: int,
                       max_parents: int = 2) -> List[int]:
    """
    Hill-climbing structure search starting from the Chow-Liu tree.

    At each step, tries:
      (a) Adding an edge   u → i  (if i doesn't yet have max_parents parents)
      (b) Removing an edge pa(i) → i
      (c) Reversing an edge pa(i) → i  to  i → pa(i)

    Stops when no single operation improves BIC.

    Parameters
    ----------
    max_parents : maximum in-degree per node (default 2, avoids CPT explosion)
    """
    mi_matrix = compute_mi_matrix(seqs, n)
    parents   = chow_liu_tree(mi_matrix, n)
    current   = _bic_score(seqs, parents, n)

    improved = True
    while improved:
        improved = False
        best_bic = current
        best_cfg = None

        for i in range(n):
            pa_i = parents[i]
            n_pa_i = sum(1 for p in parents if p != -1 and parents.index(p) == i
                         ) if False else (0 if pa_i == -1 else 1)

            # (a) Try adding an edge u→i
            if n_pa_i < max_parents and pa_i == -1:
                for u in range(n):
                    if u == i:
                        continue
                    new_p = parents[:]
                    new_p[i] = u
                    if _has_cycle(new_p, n):
                        continue
                    bic = _bic_score(seqs, new_p, n)
                    if bic > best_bic + 1e-6:
                        best_bic = bic
                        best_cfg = new_p[:]

            # (b) Try removing edge pa_i→i
            if pa_i != -1:
                new_p = parents[:]
                new_p[i] = -1
                bic = _bic_score(seqs, new_p, n)
                if bic > best_bic + 1e-6:
                    best_bic = bic
                    best_cfg = new_p[:]

            # (c) Try reversing edge pa_i→i  to  i→pa_i
            if pa_i != -1:
                new_p = parents[:]
                new_p[i]    = -1
                new_p[pa_i] = i
                if not _has_cycle(new_p, n):
                    bic = _bic_score(seqs, new_p, n)
                    if bic > best_bic + 1e-6:
                        best_bic = bic
                        best_cfg = new_p[:]

        if best_cfg is not None:
            parents  = best_cfg
            current  = best_bic
            improved = True

    return parents

# ─────────────────────────────────────────────────────────────
# Main Model Class
# ─────────────────────────────────────────────────────────────

class BayesianNetworkModel:
    """
    Bayesian Network splice site predictor.

    Structure options
    -----------------
    'chow-liu'   : optimal tree-BN (O(n²) MI computation + Kruskal)
    'greedy-bic' : hill-climbing from Chow-Liu tree, allows multi-parent nodes
    'chain'      : forced chain (= WAM structure, useful as ablation)
    List[int]    : user-supplied parents list

    Scoring
    -------
    log-odds(s) = log P(s | fg BN) − log P(s | bg BN)
    """

    def __init__(self, window: int, site: str = 'donor',
                 structure='chow-liu') -> None:
        self.window    = window
        self.site      = site
        self.structure = structure

        self.parents:            Optional[List[int]]   = None
        self.root_fg:            Optional[Dict]        = None
        self.cpt_fg:             Optional[List]        = None
        self.root_bg:            Optional[Dict]        = None
        self.cpt_bg:             Optional[List]        = None
        self.mi_matrix:          Optional[List[List[float]]] = None
        self._marginals_fg:      Optional[List[Dict]]  = None

    # ── Training ─────────────────────────────────────────────

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        """
        Learn structure from positive sequences, then fit CPTs for
        both foreground (pos) and background (neg) using the same graph.
        """
        pos = validate_seqs(pos_seqs, self.window)
        neg = validate_seqs(neg_seqs, self.window)
        if not pos:
            raise ValueError("No valid positive sequences after filtering.")

        n = self.window

        # ── Structure learning ─────────────────────────────────
        if isinstance(self.structure, list):
            self.parents = list(self.structure)

        elif self.structure == 'chain':
            self.parents = [-1] + list(range(n - 1))

        elif self.structure == 'chow-liu':
            self.mi_matrix    = compute_mi_matrix(pos, n)
            self.parents      = chow_liu_tree(self.mi_matrix, n, root=0)

        elif self.structure == 'greedy-bic':
            print("  Running BIC greedy search …")
            self.parents = greedy_bic_search(pos, n)

        else:
            raise ValueError(f"Unknown structure: {self.structure!r}")

        # ── Parameter learning ─────────────────────────────────
        self._marginals_fg           = _marginals(pos, n)
        self.root_fg, self.cpt_fg   = learn_cpts(pos, self.parents, n)
        bg_seqs = neg if neg else pos
        self.root_bg, self.cpt_bg   = learn_cpts(bg_seqs, self.parents, n)

    # ── Scoring ───────────────────────────────────────────────

    def _log_prob(self, seq: str, root_marg: Dict,
                  cpt: List) -> float:
        """log P(seq | BN)."""
        lp = 0.0
        for i in range(self.window):
            b = seq[i]
            if b not in BASE_IDX:
                return float('-inf')
            pa = self.parents[i]
            if pa == -1:
                prob = root_marg.get(b, 1e-10)
            else:
                pb   = seq[pa]
                prob = cpt[i][pb].get(b, 1e-10) if pb in BASE_IDX else 1e-10
            lp += math.log(max(prob, 1e-10))
        return lp

    def score(self, seq: str) -> float:
        """Log-odds score for a single window sequence."""
        if self.parents is None:
            raise RuntimeError("Model not trained. Call train() first.")
        seq = seq.upper()
        if len(seq) != self.window:
            raise ValueError(f"Expected length {self.window}, got {len(seq)}")
        return (self._log_prob(seq, self.root_fg, self.cpt_fg) -
                self._log_prob(seq, self.root_bg, self.cpt_bg))

    def score_batch(self, seqs: List[str]) -> List[float]:
        return [self.score(s) for s in seqs]

    def predict(self, seq: str, threshold: float = 0.0) -> int:
        return int(self.score(seq) >= threshold)

    def predict_batch(self, seqs: List[str],
                      threshold: float = 0.0) -> List[int]:
        return [self.predict(s, threshold) for s in seqs]

    # ── Analysis ──────────────────────────────────────────────

    def information_content(self) -> List[float]:
        """
        Per-position IC (bits) against uniform background.
        For non-root nodes: average over parent states.
        """
        if self.parents is None or self.root_fg is None:
            raise RuntimeError("Not trained.")
        bg = {b: 0.25 for b in BASES}
        ic = []
        for i in range(self.window):
            pa = self.parents[i]
            if pa == -1:
                ic_i = sum(
                    self.root_fg[b] * math.log2(max(self.root_fg[b], 1e-10) / bg[b])
                    for b in BASES if self.root_fg[b] > 0
                )
            else:
                ic_i = 0.0
                for pa_b in BASES:
                    for ch_b in BASES:
                        p = self.cpt_fg[i][pa_b][ch_b]
                        if p > 0:
                            ic_i += 0.25 * p * math.log2(p / bg[ch_b])
            ic.append(ic_i)
        return ic

    def non_adjacent_edges(self) -> List[Tuple[int, int]]:
        """Return list of (child, parent) pairs where |child-parent| > 1."""
        if self.parents is None:
            return []
        return [(i, p) for i, p in enumerate(self.parents)
                if p != -1 and abs(i - p) > 1]

    def print_mi_matrix(self) -> None:
        """Print the mutual information matrix as an ASCII heatmap."""
        if self.mi_matrix is None:
            print("MI matrix not available (only for Chow-Liu/greedy-BIC).")
            return
        n = self.window
        print(f"\nMutual Information matrix ({self.site} site, window={n})")
        print("    " + "".join(f" {j:5d}" for j in range(n)))
        print("    " + "─" * (6 * n))
        for i in range(n):
            row = f"{i:>3} |"
            for j in range(n):
                v = self.mi_matrix[i][j]
                if i == j:
                    row += "   ─  "
                elif v >= 0.3:
                    row += f" {v:5.3f}"   # high MI
                else:
                    row += f" {v:5.3f}"
            print(row)

    def print_summary(self) -> None:
        """Full summary: structure, non-adjacent edges, IC per position."""
        if self.parents is None:
            print("Not trained.")
            return
        root = next(i for i, p in enumerate(self.parents) if p == -1)
        non_adj = self.non_adjacent_edges()

        print(f"\nBayesian Network — {self.site} site  (window={self.window}, "
              f"structure={self.structure})")
        print("=" * 62)
        print(f"Root: position {root}")
        print(f"Edges learned: {self.window - 1}")
        print(f"Non-adjacent edges (long-range): {len(non_adj)}  "
              + (str([(f'{c}←{p}', f'gap={abs(c-p)}') for c,p in non_adj])
                 if non_adj else "(none)"))

        ic = self.information_content()
        print(f"\n{'Pos':>4}  {'Parent':>8}  {'IC (bits)':>10}  {'Edge type'}")
        print("─" * 48)
        for i, (pa, ic_val) in enumerate(zip(self.parents, ic)):
            if pa == -1:
                pa_str   = "root"
                edge_typ = "—"
            elif abs(i - pa) == 1:
                pa_str   = str(pa)
                edge_typ = "adjacent"
            else:
                pa_str   = str(pa)
                edge_typ = f"LONG-RANGE gap={abs(i-pa)}"
            print(f"{i:>4}  {pa_str:>8}  {ic_val:>10.4f}  {edge_typ}")
        print("=" * 62)

# ─────────────────────────────────────────────────────────────
# Genome Scanner
# ─────────────────────────────────────────────────────────────

class BNScanner:
    """
    Slide a BayesianNetworkModel over a genomic sequence.

    site='donor'    → looks for GT at canonical position
    site='acceptor' → looks for AG
    """
    DONOR_OFFSET    = 3   # GT starts at position 3 in 9-bp window
    ACCEPTOR_OFFSET = 20  # AG starts at position 20 in 23-bp window

    def __init__(self, model: BayesianNetworkModel,
                 threshold: float = 0.0) -> None:
        self.model     = model
        self.threshold = threshold
        self.site      = model.site
        self.offset    = (self.DONOR_OFFSET if self.site == 'donor'
                          else self.ACCEPTOR_OFFSET)
        self.dinuc     = 'GT' if self.site == 'donor' else 'AG'

    def scan(self, genome: str) -> List[Tuple[int, float]]:
        """
        Return (dinucleotide_position, score) for every predicted site.
        Filters by canonical GT/AG dinucleotide before scoring.
        """
        genome = genome.upper()
        w      = self.model.window
        off    = self.offset
        hits: List[Tuple[int, float]] = []
        for i in range(off, len(genome) - w + off + 1):
            win = genome[i - off: i - off + w]
            if len(win) != w:
                continue
            if win[off: off + 2] != self.dinuc:
                continue
            sc = self.model.score(win)
            if sc >= self.threshold:
                hits.append((i, sc))
        return hits

# ─────────────────────────────────────────────────────────────
# Baseline models (WMM and WAM, for comparison)
# ─────────────────────────────────────────────────────────────

class _WMMModel:
    """Zero-order position weight matrix (baseline)."""

    def __init__(self, window: int) -> None:
        self.window   = window
        self.log_odds: Optional[List[Dict[str, float]]] = None

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        pos = validate_seqs(pos_seqs, self.window)
        n   = self.window
        bg  = empirical_bg(neg_seqs) if neg_seqs else {b: 0.25 for b in BASES}
        cts = [{b: PSEUDOCOUNT for b in BASES} for _ in range(n)]
        for seq in pos:
            for i, c in enumerate(seq):
                cts[i][c] += 1
        self.log_odds = []
        for pos_ct in cts:
            total = sum(pos_ct.values())
            freq  = {b: pos_ct[b] / total for b in BASES}
            self.log_odds.append(
                {b: math.log(max(freq[b], 1e-10) / max(bg[b], 1e-10))
                 for b in BASES}
            )

    def score(self, seq: str) -> float:
        seq = seq.upper()
        return sum(self.log_odds[i].get(c, 0.0)
                   for i, c in enumerate(seq) if c in BASE_IDX)

    def score_batch(self, seqs: List[str]) -> List[float]:
        return [self.score(s) for s in seqs]


class _WAMModel:
    """First-order Markov (chain BN = WAM) for comparison."""

    def __init__(self, window: int) -> None:
        self.window = window
        self._bn    = BayesianNetworkModel(window, structure='chain')

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        self._bn.train(pos_seqs, neg_seqs)

    def score(self, seq: str) -> float:
        return self._bn.score(seq)

    def score_batch(self, seqs: List[str]) -> List[float]:
        return self._bn.score_batch(seqs)

# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(t == p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == p == 0 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                sensitivity=sens, specificity=spec,
                precision=prec, F1=f1, MCC=mcc)

def roc_auc(y_true: List[int], scores: List[float],
            n_thr: int = 200) -> Tuple[List[float], List[float], float]:
    thresholds = sorted(set(scores), reverse=True)
    step       = max(1, len(thresholds) // n_thr)
    thresholds = thresholds[::step]
    fpr, tpr   = [0.0], [0.0]
    for thr in thresholds:
        pred = [int(s >= thr) for s in scores]
        m    = evaluate(y_true, pred)
        fpr.append(1.0 - m['specificity'])
        tpr.append(m['sensitivity'])
    fpr.append(1.0); tpr.append(1.0)
    auc = sum((fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
              for i in range(1, len(fpr)))
    return fpr, tpr, auc

# ─────────────────────────────────────────────────────────────
# Synthetic dataset (same generator as wam_splice_site.py)
# ─────────────────────────────────────────────────────────────

def _rand(weights: Dict[str, float]) -> str:
    r = random.random()
    cum = 0.0
    for b, w in weights.items():
        cum += w
        if r < cum:
            return b
    return 'A'

def make_donor_positive(n: int = 500) -> List[str]:
    """
    Synthetic 9-bp donor windows with biological covariance:
    - Strong GT at positions 3-4 (100%)
    - Purines preferred at -3 (pos 0) and +3 (pos 5)
    - A preferred at +4 (pos 6)
    - Covariation: if pos 0 = A then pos 2 biased toward G (mimics real correlation)
    """
    seqs = []
    for _ in range(n):
        p0 = _rand({'A': 0.35, 'G': 0.35, 'C': 0.15, 'T': 0.15})
        p1 = _rand({'A': 0.45, 'G': 0.35, 'C': 0.10, 'T': 0.10})
        # p2 correlates with p0 (long-range)
        p2 = ('G' if p0 in ('A', 'G') and random.random() < 0.7
              else _rand({'A': 0.2, 'G': 0.5, 'C': 0.15, 'T': 0.15}))
        p3, p4 = 'G', 'T'            # canonical GT
        p5 = _rand({'A': 0.65, 'G': 0.15, 'C': 0.10, 'T': 0.10})
        p6 = _rand({'A': 0.70, 'G': 0.10, 'C': 0.10, 'T': 0.10})
        p7 = _rand({'A': 0.10, 'G': 0.65, 'C': 0.10, 'T': 0.15})
        # p8 correlates with p5 (long-range)
        p8 = ('T' if p5 == 'A' and random.random() < 0.75
              else _rand({'A': 0.1, 'G': 0.1, 'C': 0.1, 'T': 0.7}))
        seqs.append(''.join([p0, p1, p2, p3, p4, p5, p6, p7, p8]))
    return seqs

def make_donor_negative(n: int = 500) -> List[str]:
    """Random sequences with GT at positions 3-4 (pseudo-decoys)."""
    seqs = []
    while len(seqs) < n:
        s = list(''.join(random.choices(list(BASES), k=9)))
        s[3], s[4] = 'G', 'T'
        seqs.append(''.join(s))
    return seqs

# ─────────────────────────────────────────────────────────────
# Comparison runner
# ─────────────────────────────────────────────────────────────

def compare_models(train_pos: List[str], train_neg: List[str],
                   test_pos: List[str],  test_neg: List[str],
                   window: int = DONOR_WINDOW,
                   threshold: float = 0.0) -> None:
    """Train WMM, WAM, BN (Chow-Liu), and BN (greedy-BIC); compare on test set."""

    models = [
        ("WMM",          _WMMModel(window)),
        ("WAM",          _WAMModel(window)),
        ("BN Chow-Liu",  BayesianNetworkModel(window, site='donor',
                                               structure='chow-liu')),
        ("BN greedy-BIC",BayesianNetworkModel(window, site='donor',
                                               structure='greedy-bic')),
    ]

    print("\nTraining models …")
    for name, m in models:
        print(f"  {name} …", end=" ", flush=True)
        m.train(train_pos, train_neg)
        print("done")

    test_seqs   = test_pos  + test_neg
    test_labels = [1]*len(test_pos) + [0]*len(test_neg)

    print("\n" + "=" * 70)
    print("  WMM / WAM / BN Chow-Liu / BN greedy-BIC  — Donor Site Prediction")
    print("=" * 70)
    hdr = f"{'Metric':<20}" + "".join(f"{n:>15}" for n, _ in models)
    print(hdr)
    print("─" * 80)

    result_rows: List[List] = []
    aucs = []
    for name, m in models:
        scores = m.score_batch(test_seqs)
        preds  = [int(s >= threshold) for s in scores]
        met    = evaluate(test_labels, preds)
        _, _, auc = roc_auc(test_labels, scores)
        result_rows.append((met, auc))
        aucs.append(auc)

    metrics_to_show = [
        ("Sensitivity",   "sensitivity"),
        ("Specificity",   "specificity"),
        ("Precision",     "precision"),
        ("F1 score",      "F1"),
        ("MCC",           "MCC"),
        ("AUC-ROC",       None),
    ]
    for label, key in metrics_to_show:
        if key is None:
            vals = aucs
        else:
            vals = [row[0][key] for row in result_rows]
        best = max(vals)
        row  = f"  {label:<18}"
        for v in vals:
            row += f"{v:>13.4f}" + ("◀" if abs(v - best) < 1e-6 else " ")
        print(row)

    print("─" * 80)
    print("  ◀ = best in row")
    print("=" * 70)

    # Show BN structure details
    bn_cl = models[2][1]
    bn_cl.print_summary()

    non_adj = bn_cl.non_adjacent_edges()
    if non_adj:
        print(f"\n  Long-range edges discovered by Chow-Liu (not captured by WAM):")
        for child, parent in non_adj:
            mi_val = bn_cl.mi_matrix[child][parent] if bn_cl.mi_matrix else float('nan')
            print(f"    pos {parent} → pos {child}  "
                  f"gap={abs(child-parent)}  MI={mi_val:.4f} nats")

# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

def demo() -> None:
    """Self-contained demo using synthetic data."""
    random.seed(42)

    print("=" * 60)
    print("  Bayesian Network Splice Site Predictor — Demo")
    print("=" * 60)
    print("Generating synthetic donor-site dataset with")
    print("embedded long-range covariation (pos 0↔2, pos 5↔8) …\n")

    all_pos = make_donor_positive(1000)
    all_neg = make_donor_negative(1000)
    split   = 800

    compare_models(
        train_pos=all_pos[:split],  train_neg=all_neg[:split],
        test_pos=all_pos[split:],   test_neg=all_neg[split:],
    )

    # ── MI matrix heatmap ──────────────────────────────────────
    bn = BayesianNetworkModel(window=DONOR_WINDOW, site='donor',
                               structure='chow-liu')
    bn.train(all_pos[:split], all_neg[:split])
    bn.print_mi_matrix()

    # ── Genome scan ────────────────────────────────────────────
    print("\n── Genome scan demo ─────────────────────────────────────")
    scanner = BNScanner(bn, threshold=1.5)
    genome  = ("ATCGATCGATCG"
               "CAGGTAAGTATCG"   # planted donor 1 @ offset 12
               "GCATCGATCGATCG"
               "AAGGTAAGTGCTA"   # planted donor 2 @ offset 39
               "TTTGCATCGATCG")
    hits = scanner.scan(genome)
    print(f"Genome: {len(genome)} bp | threshold={scanner.threshold}")
    for pos, sc in hits:
        ctx = genome[max(0, pos-2): pos+5]
        print(f"  pos={pos:>4}  score={sc:+.3f}  context: …{ctx}…")
    if not hits:
        print("  (none above threshold)")


if __name__ == '__main__':
    demo()