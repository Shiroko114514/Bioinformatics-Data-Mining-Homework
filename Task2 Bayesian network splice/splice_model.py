import math
from typing import List, Dict, Optional, Tuple
from splice_utils import _marginals, learn_cpts, compute_mi_matrix, chow_liu_tree, validate_seqs, BASES, BASE_IDX

class BayesianNetworkModel:
    def __init__(self, window: int, site: str = 'donor', structure='chow-liu') -> None:
        self.window = window
        self.site = site
        self.structure = structure
        self.parents: Optional[List[int]] = None
        self.root_fg: Optional[Dict] = None
        self.cpt_fg: Optional[List] = None
        self.root_bg: Optional[Dict] = None
        self.cpt_bg: Optional[List] = None
        self.mi_matrix: Optional[List[List[float]]] = None
        self._marginals_fg: Optional[List[Dict]] = None

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        pos = validate_seqs(pos_seqs, self.window)
        neg = validate_seqs(neg_seqs, self.window)
        if not pos:
            raise ValueError("No valid positive sequences after filtering.")

        n = self.window

        if isinstance(self.structure, list):
            self.parents = list(self.structure)
        elif self.structure == 'chain':
            self.parents = [-1] + list(range(n - 1))
        elif self.structure == 'chow-liu':
            self.mi_matrix = compute_mi_matrix(pos, n)
            self.parents = chow_liu_tree(self.mi_matrix, n, root=0)
        elif self.structure == 'greedy-bic':
            raise NotImplementedError('greedy-bic in separate module')
        else:
            raise ValueError(f"Unknown structure: {self.structure!r}")

        self._marginals_fg = _marginals(pos, n)
        self.root_fg, self.cpt_fg = learn_cpts(pos, self.parents, n)
        bg_seqs = neg if neg else pos
        self.root_bg, self.cpt_bg = learn_cpts(bg_seqs, self.parents, n)

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

    def score(self, seq: str) -> float:
        if self.parents is None:
            raise RuntimeError("Model not trained. Call train() first.")
        seq = seq.upper()
        if len(seq) != self.window:
            raise ValueError(f"Expected length {self.window}, got {len(seq)}")
        return (self._log_prob(seq, self.root_fg, self.cpt_fg)
                - self._log_prob(seq, self.root_bg, self.cpt_bg))

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
