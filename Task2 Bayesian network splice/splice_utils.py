import math
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

BASES = ('A', 'C', 'G', 'T')
BASE_IDX = {b: i for i, b in enumerate(BASES)}
PSEUDOCOUNT = 0.5

DONOR_WINDOW = 9
ACCEPTOR_WINDOW = 23


def validate_seqs(seqs: List[str], window: int) -> List[str]:
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


def _marginals(seqs: List[str], n: int) -> List[Dict[str, float]]:
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
    counts: Dict[Tuple[str, str], float] = defaultdict(lambda: PSEUDOCOUNT)
    for seq in seqs:
        a, b = seq[i], seq[j]
        if a in BASE_IDX and b in BASE_IDX:
            counts[(a, b)] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def mutual_information(seqs: List[str], i: int, j: int,
                       marginals: List[Dict[str, float]]) -> float:
    joint = _joint(seqs, i, j)
    mi = 0.0
    for a in BASES:
        for b in BASES:
            pab = joint.get((a, b), 1e-10)
            pa = marginals[i][a]
            pb = marginals[j][b]
            if pab > 0 and pa > 0 and pb > 0:
                mi += pab * math.log(pab / (pa * pb))
    return max(mi, 0.0)


def compute_mi_matrix(seqs: List[str], n: int) -> List[List[float]]:
    marginals = _marginals(seqs, n)
    mi = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m = mutual_information(seqs, i, j, marginals)
            mi[i][j] = m
            mi[j][i] = m
    return mi


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

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


def chow_liu_tree(mi_matrix: List[List[float]], n: int,
                   root: int = 0) -> List[int]:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((mi_matrix[i][j], i, j))
    edges.sort(reverse=True)

    uf = _UnionFind(n)
    adj: Dict[int, List[int]] = defaultdict(list)
    for mi_val, u, v in edges:
        if uf.union(u, v):
            adj[u].append(v)
            adj[v].append(u)

    parents = [-1] * n
    visited = [False] * n
    queue = deque([root])
    visited[root] = True
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                parents[nb] = node
                queue.append(nb)
    return parents


def learn_cpts(seqs: List[str], parents: List[int], n: int
               ) -> Tuple[Dict[str, float], List[Optional[Dict]]]:
    root = next(i for i, p in enumerate(parents) if p == -1)
    r_counts = {b: PSEUDOCOUNT for b in BASES}
    for seq in seqs:
        b = seq[root]
        if b in BASE_IDX:
            r_counts[b] += 1
    total_r = sum(r_counts.values())
    root_marginal = {b: r_counts[b] / total_r for b in BASES}

    cpt_list: List[Optional[Dict]] = [None] * n
    for i in range(n):
        p = parents[i]
        if p == -1:
            continue
        cond = {pa: {ch: PSEUDOCOUNT for ch in BASES} for pa in BASES}
        for seq in seqs:
            pa_b, ch_b = seq[p], seq[i]
            if pa_b in BASE_IDX and ch_b in BASE_IDX:
                cond[pa_b][ch_b] += 1
        cpt = {}
        for pa_b in BASES:
            total = sum(cond[pa_b].values())
            cpt[pa_b] = {ch_b: cond[pa_b][ch_b] / total for ch_b in BASES}
        cpt_list[i] = cpt

    return root_marginal, cpt_list
