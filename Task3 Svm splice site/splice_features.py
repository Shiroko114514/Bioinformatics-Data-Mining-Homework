import itertools
import math
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from splice_utils import BASE_IDX, BASES, PSEUDOCOUNT


class FeatureExtractor:
    """
    Transform raw nucleotide windows into numeric feature vectors.

    Supported feature types
    -----------------------
    'one_hot'     : flattened one-hot, shape (4L,)
    'kmer2'       : dinucleotide frequencies, shape (16,)
    'kmer3'       : trinucleotide frequencies, shape (64,)
    'dinuc_pos'   : position-specific dinucleotide one-hot, shape (16*(L-1),)
    'pwm'         : per-position WMM log-odds trained on pos set, shape (L,)
    'chi2_pairs'  : chi2-selected non-local pair one-hot, shape (16 * n_pairs,)
    'ebn_llr'     : EBN-style log-likelihood ratio score, shape (1,)
    'combined'    : concatenation of all of the above
    """

    ALL_FEATURES = ("one_hot", "kmer2", "kmer3", "dinuc_pos", "pwm", "chi2_pairs", "ebn_llr")

    def __init__(
        self,
        window: int,
        features: Union[str, List[str]] = "combined",
        dependency_threshold: float = 6.0,
        max_dependency_pairs: int = 16,
        ebn_max_parents: int = 2,
    ) -> None:
        self.window = window
        self.dependency_threshold = float(dependency_threshold)
        self.max_dependency_pairs = max(0, int(max_dependency_pairs))
        self.ebn_max_parents = max(1, int(ebn_max_parents))
        if features == "combined":
            self.features = list(self.ALL_FEATURES)
        elif isinstance(features, str):
            self.features = [features]
        else:
            self.features = list(features)

        self._kmers2 = ["".join(p) for p in itertools.product(BASES, repeat=2)]
        self._kmers3 = ["".join(p) for p in itertools.product(BASES, repeat=3)]
        self._kmer2_idx = {k: i for i, k in enumerate(self._kmers2)}
        self._kmer3_idx = {k: i for i, k in enumerate(self._kmers3)}

        self._pwm_lo: Optional[List[Dict[str, float]]] = None
        self._chi2_pairs: List[Tuple[int, int]] = []
        self._chi2_matrix: Optional[np.ndarray] = None

        self._ebn_order: Optional[List[int]] = None
        self._ebn_parents: Dict[int, List[int]] = {}
        self._ebn_pos_prob: Dict[int, Dict[Tuple[int, ...], Dict[str, float]]] = {}
        self._ebn_neg_prob: Dict[int, Dict[Tuple[int, ...], Dict[str, float]]] = {}

    def _compute_chi2_matrix(self, seqs: List[str]) -> np.ndarray:
        n = self.window
        chi2 = np.zeros((n, n), dtype=np.float32)
        if not seqs:
            return chi2

        total = len(seqs)
        for i in range(n):
            for j in range(i + 1, n):
                table = np.zeros((4, 4), dtype=np.float64)
                for s in seqs:
                    ai = BASE_IDX.get(s[i], -1)
                    bj = BASE_IDX.get(s[j], -1)
                    if ai >= 0 and bj >= 0:
                        table[ai, bj] += 1.0

                if table.sum() == 0.0:
                    continue
                row_sum = table.sum(axis=1)
                col_sum = table.sum(axis=0)
                score = 0.0
                for r in range(4):
                    for c in range(4):
                        expected = (row_sum[r] * col_sum[c]) / max(float(total), 1.0)
                        if expected > 1e-12:
                            diff = table[r, c] - expected
                            score += (diff * diff) / expected
                chi2[i, j] = score
                chi2[j, i] = score
        return chi2

    def _select_dependency_pairs(self, chi2: np.ndarray) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int, float]] = []
        for i in range(self.window):
            for j in range(i + 1, self.window):
                sc = float(chi2[i, j])
                if sc >= self.dependency_threshold:
                    pairs.append((i, j, sc))

        if not pairs:
            for i in range(self.window):
                for j in range(i + 1, self.window):
                    pairs.append((i, j, float(chi2[i, j])))

        pairs.sort(key=lambda x: x[2], reverse=True)
        keep = pairs[: self.max_dependency_pairs]
        return [(i, j) for i, j, _ in keep]

    def _build_ebn_structure(self, chi2: np.ndarray) -> Tuple[List[int], Dict[int, List[int]]]:
        n = self.window
        # Root is the position with max total dependency strength.
        dep_strength = [float(np.sum(chi2[i])) for i in range(n)]
        root = int(np.argmax(np.array(dep_strength))) if dep_strength else 0

        neigh = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if float(chi2[i, j]) >= self.dependency_threshold:
                    neigh[i].append(j)
                    neigh[j].append(i)

        # Ensure every node is reachable by adding strongest missing links.
        for i in range(n):
            if neigh[i]:
                continue
            cand = sorted(((j, float(chi2[i, j])) for j in range(n) if j != i), key=lambda x: x[1], reverse=True)
            if cand:
                j = cand[0][0]
                neigh[i].append(j)
                neigh[j].append(i)

        order: List[int] = []
        seen = set()
        q: deque[int] = deque([root])
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            order.append(u)
            nxt = sorted(neigh[u], key=lambda v: float(chi2[u, v]), reverse=True)
            for v in nxt:
                if v not in seen:
                    q.append(v)

        if len(order) < n:
            rest = [i for i in range(n) if i not in seen]
            rest.sort(key=lambda i: dep_strength[i], reverse=True)
            order.extend(rest)

        parents: Dict[int, List[int]] = {order[0]: []}
        for idx in range(1, len(order)):
            node = order[idx]
            prev_nodes = order[:idx]
            ranked_prev = sorted(prev_nodes, key=lambda p: float(chi2[node, p]), reverse=True)
            parents[node] = ranked_prev[: self.ebn_max_parents]
        return order, parents

    def _fit_ebn_class(
        self,
        seqs: List[str],
        order: List[int],
        parents: Dict[int, List[int]],
    ) -> Dict[int, Dict[Tuple[int, ...], Dict[str, float]]]:
        out: Dict[int, Dict[Tuple[int, ...], Dict[str, float]]] = {}
        for node in order:
            node_parents = parents.get(node, [])
            by_parent: Dict[Tuple[int, ...], Counter[str]] = {}
            for s in seqs:
                key = tuple(BASE_IDX[s[p]] for p in node_parents)
                by_parent.setdefault(key, Counter())
                by_parent[key][s[node]] += 1

            prob_map: Dict[Tuple[int, ...], Dict[str, float]] = {}
            for key, ct in by_parent.items():
                total = float(sum(ct.values())) + 4.0
                prob_map[key] = {
                    b: (float(ct.get(b, 0.0)) + 1.0) / total
                    for b in BASES
                }
            out[node] = prob_map
        return out

    def _ebn_log_prob(self, seq: str, cls_prob: Dict[int, Dict[Tuple[int, ...], Dict[str, float]]]) -> float:
        if self._ebn_order is None:
            raise RuntimeError("Call fit() before extracting 'ebn_llr' features.")
        lp = 0.0
        for node in self._ebn_order:
            node_parents = self._ebn_parents.get(node, [])
            key = tuple(BASE_IDX[seq[p]] for p in node_parents)
            prob_map = cls_prob.get(node, {})
            cond = prob_map.get(key)
            if cond is None:
                # Backoff to uniform when this parent state is unseen.
                p = 0.25
            else:
                p = cond.get(seq[node], 0.25)
            lp += math.log(max(p, 1e-10))
        return lp

    def fit(self, pos_seqs: List[str], neg_seqs: Optional[List[str]] = None) -> "FeatureExtractor":
        n = self.window
        counts = [{b: PSEUDOCOUNT for b in BASES} for _ in range(n)]
        for seq in pos_seqs:
            for i, c in enumerate(seq[:n]):
                if c in BASE_IDX:
                    counts[i][c] += 1
        bg = {b: 0.25 for b in BASES}
        self._pwm_lo = []
        for pos_ct in counts:
            total = sum(pos_ct.values())
            self._pwm_lo.append({b: math.log(max(pos_ct[b] / total, 1e-10) / bg[b]) for b in BASES})

        need_dep = any(ft in {"chi2_pairs", "ebn_llr"} for ft in self.features)
        if need_dep:
            self._chi2_matrix = self._compute_chi2_matrix(pos_seqs)
            self._chi2_pairs = self._select_dependency_pairs(self._chi2_matrix)

        if "ebn_llr" in self.features:
            if neg_seqs is None or not neg_seqs:
                raise ValueError("'ebn_llr' feature requires negative sequences during fit().")
            if self._chi2_matrix is None:
                self._chi2_matrix = self._compute_chi2_matrix(pos_seqs)
            order, parents = self._build_ebn_structure(self._chi2_matrix)
            self._ebn_order = order
            self._ebn_parents = parents
            self._ebn_pos_prob = self._fit_ebn_class(pos_seqs, order, parents)
            self._ebn_neg_prob = self._fit_ebn_class(neg_seqs, order, parents)
        return self

    def _one_hot(self, seq: str) -> np.ndarray:
        vec = np.zeros(4 * self.window, dtype=np.float32)
        for i, c in enumerate(seq):
            if c in BASE_IDX:
                vec[i * 4 + BASE_IDX[c]] = 1.0
        return vec

    def _kmer_freq(self, seq: str, k: int) -> np.ndarray:
        idx_map = self._kmer2_idx if k == 2 else self._kmer3_idx
        n_kmers = len(idx_map)
        vec = np.zeros(n_kmers, dtype=np.float32)
        count = 0
        for i in range(len(seq) - k + 1):
            km = seq[i : i + k]
            if km in idx_map:
                vec[idx_map[km]] += 1
                count += 1
        if count > 0:
            vec /= count
        return vec

    def _dinuc_pos(self, seq: str) -> np.ndarray:
        size = (self.window - 1) * 16
        vec = np.zeros(size, dtype=np.float32)
        for i in range(self.window - 1):
            a, b = seq[i], seq[i + 1]
            if a in BASE_IDX and b in BASE_IDX:
                di_idx = BASE_IDX[a] * 4 + BASE_IDX[b]
                vec[i * 16 + di_idx] = 1.0
        return vec

    def _pwm_scores(self, seq: str) -> np.ndarray:
        if self._pwm_lo is None:
            raise RuntimeError("Call fit() before extracting 'pwm' features.")
        vec = np.zeros(self.window, dtype=np.float32)
        for i, c in enumerate(seq[: self.window]):
            if c in BASE_IDX:
                vec[i] = float(self._pwm_lo[i].get(c, 0.0))
        return vec

    def _chi2_pair_features(self, seq: str) -> np.ndarray:
        if not self._chi2_pairs:
            return np.zeros(0, dtype=np.float32)
        vec = np.zeros(len(self._chi2_pairs) * 16, dtype=np.float32)
        for k, (i, j) in enumerate(self._chi2_pairs):
            a = BASE_IDX.get(seq[i], -1)
            b = BASE_IDX.get(seq[j], -1)
            if a >= 0 and b >= 0:
                idx = k * 16 + a * 4 + b
                vec[idx] = 1.0
        return vec

    def _ebn_llr(self, seq: str) -> np.ndarray:
        if self._ebn_order is None:
            raise RuntimeError("Call fit() before extracting 'ebn_llr' features.")
        ll_pos = self._ebn_log_prob(seq, self._ebn_pos_prob)
        ll_neg = self._ebn_log_prob(seq, self._ebn_neg_prob)
        # Paper-style log-likelihood ratio: log P(seq|false) / P(seq|true)
        return np.array([ll_neg - ll_pos], dtype=np.float32)

    def transform_one(self, seq: str) -> np.ndarray:
        parts = []
        for ft in self.features:
            if ft == "one_hot":
                parts.append(self._one_hot(seq))
            elif ft == "kmer2":
                parts.append(self._kmer_freq(seq, 2))
            elif ft == "kmer3":
                parts.append(self._kmer_freq(seq, 3))
            elif ft == "dinuc_pos":
                parts.append(self._dinuc_pos(seq))
            elif ft == "pwm":
                parts.append(self._pwm_scores(seq))
            elif ft == "chi2_pairs":
                parts.append(self._chi2_pair_features(seq))
            elif ft == "ebn_llr":
                parts.append(self._ebn_llr(seq))
            else:
                raise ValueError(f"Unknown feature type: {ft!r}")
        return np.concatenate(parts)

    def transform(self, seqs: List[str]) -> np.ndarray:
        return np.stack([self.transform_one(s) for s in seqs])

    @property
    def feature_dim(self) -> int:
        dim = 0
        for ft in self.features:
            if ft == "one_hot":
                dim += 4 * self.window
            elif ft == "kmer2":
                dim += 16
            elif ft == "kmer3":
                dim += 64
            elif ft == "dinuc_pos":
                dim += 16 * (self.window - 1)
            elif ft == "pwm":
                dim += self.window
            elif ft == "chi2_pairs":
                dim += 16 * len(self._chi2_pairs)
            elif ft == "ebn_llr":
                dim += 1
        return dim

    def feature_names(self) -> List[str]:
        names = []
        for ft in self.features:
            if ft == "one_hot":
                for i in range(self.window):
                    for b in BASES:
                        names.append(f"oh_pos{i}_{b}")
            elif ft == "kmer2":
                names += [f"k2_{k}" for k in self._kmers2]
            elif ft == "kmer3":
                names += [f"k3_{k}" for k in self._kmers3]
            elif ft == "dinuc_pos":
                for i in range(self.window - 1):
                    for a in BASES:
                        for b in BASES:
                            names.append(f"dn_pos{i}_{a}{b}")
            elif ft == "pwm":
                names += [f"pwm_pos{i}" for i in range(self.window)]
            elif ft == "chi2_pairs":
                for i, j in self._chi2_pairs:
                    for a in BASES:
                        for b in BASES:
                            names.append(f"chi2_pair_{i}_{j}_{a}{b}")
            elif ft == "ebn_llr":
                names.append("ebn_llr_false_vs_true")
        return names
