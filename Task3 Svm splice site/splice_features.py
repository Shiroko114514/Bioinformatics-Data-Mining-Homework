import itertools
import math
from typing import Dict, List, Optional, Union

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
    'combined'    : concatenation of all of the above
    """

    ALL_FEATURES = ("one_hot", "kmer2", "kmer3", "dinuc_pos", "pwm")

    def __init__(self, window: int, features: Union[str, List[str]] = "combined") -> None:
        self.window = window
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

    def fit(self, pos_seqs: List[str]) -> "FeatureExtractor":
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
        return names
