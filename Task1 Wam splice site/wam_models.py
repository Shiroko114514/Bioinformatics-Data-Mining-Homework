import math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

BASES = ("A", "C", "G", "T")
BASE_IDX = {b: i for i, b in enumerate(BASES)}

DONOR_WINDOW = 9
ACCEPTOR_WINDOW = 23
PSEUDOCOUNT = 0.5


def _uniform_bg() -> Dict[str, float]:
    return {b: 0.25 for b in BASES}


def _empirical_bg(seqs: List[str]) -> Dict[str, float]:
    counts = defaultdict(int)
    total = 0
    for seq in seqs:
        for ch in seq.upper():
            if ch in BASE_IDX:
                counts[ch] += 1
                total += 1
    if total == 0:
        return _uniform_bg()
    return {b: counts[b] / total for b in BASES}


def _validate_seq(seq: str, length: int) -> Optional[str]:
    seq = seq.upper().strip()
    if len(seq) != length:
        return None
    if any(ch not in BASE_IDX for ch in seq):
        return None
    return seq


def _chi2_matrix(seqs: List[str], window: int) -> List[List[float]]:
    chi2 = [[0.0 for _ in range(window)] for _ in range(window)]
    if not seqs:
        return chi2

    total = float(len(seqs))
    for i in range(window):
        for j in range(i + 1, window):
            table = [[0.0 for _ in BASES] for _ in BASES]
            for seq in seqs:
                ai = BASE_IDX.get(seq[i], -1)
                bj = BASE_IDX.get(seq[j], -1)
                if ai >= 0 and bj >= 0:
                    table[ai][bj] += 1.0

            row_sum = [sum(table[r]) for r in range(4)]
            col_sum = [sum(table[r][c] for r in range(4)) for c in range(4)]

            score = 0.0
            for r in range(4):
                for c in range(4):
                    expected = (row_sum[r] * col_sum[c]) / total
                    if expected > 1e-12:
                        diff = table[r][c] - expected
                        score += (diff * diff) / expected
            chi2[i][j] = score
            chi2[j][i] = score
    return chi2


def _select_dependency_pairs(
    chi2: List[List[float]],
    threshold: float,
    max_pairs: int,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int, float]] = []
    n = len(chi2)
    for i in range(n):
        for j in range(i + 1, n):
            score = chi2[i][j]
            if score >= threshold:
                pairs.append((i, j, score))

    if not pairs:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, chi2[i][j]))

    pairs.sort(key=lambda item: item[2], reverse=True)
    return [(i, j) for i, j, _ in pairs[:max_pairs]]


def _pair_index(a: str, b: str) -> int:
    return BASE_IDX[a] * 4 + BASE_IDX[b]


class WMMModel:
    def __init__(self, window: int, site: str = "donor"):
        self.window = window
        self.site = site
        self.pwm_fg: Optional[List[Dict[str, float]]] = None
        self.log_odds: Optional[List[Dict[str, float]]] = None
        self.bg: Dict[str, float] = _uniform_bg()

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        valid_pos = [_validate_seq(s, self.window) for s in pos_seqs]
        valid_pos = [s for s in valid_pos if s]

        self.bg = _empirical_bg(neg_seqs)

        counts = [{b: PSEUDOCOUNT for b in BASES} for _ in range(self.window)]
        for seq in valid_pos:
            for i, ch in enumerate(seq):
                counts[i][ch] += 1

        self.pwm_fg = []
        for pos_counts in counts:
            total = sum(pos_counts.values())
            self.pwm_fg.append({b: pos_counts[b] / total for b in BASES})

        self.log_odds = []
        for i, freq in enumerate(self.pwm_fg):
            lo = {b: math.log(freq[b] / max(self.bg[b], 1e-9)) for b in BASES}
            self.log_odds.append(lo)

    def _validate_window(self, seq: str) -> bool:
        return bool(_validate_seq(seq, self.window))

    def score(self, seq: str) -> float:
        seq = seq.upper()
        if self.log_odds is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        if not self._validate_window(seq):
            return float("-inf")
        return sum(self.log_odds[i][ch] for i, ch in enumerate(seq) if ch in BASE_IDX)

    def score_batch(self, seqs: List[str]) -> List[float]:
        return [self.score(s) for s in seqs]


class WAMModel:
    def __init__(self, window: int, site: str = "donor"):
        self.window = window
        self.site = site
        self.wam_fg: Optional[List[Dict[str, Dict[str, float]]]] = None
        self.first_fg: Optional[Dict[str, float]] = None
        self.bg: Dict[str, float] = _uniform_bg()

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        valid = [_validate_seq(s, self.window) for s in pos_seqs]
        valid = [s for s in valid if s]
        if not valid:
            raise ValueError("No valid positive sequences after filtering.")

        self.bg = _empirical_bg(neg_seqs) if neg_seqs else _uniform_bg()

        first_counts = {b: PSEUDOCOUNT for b in BASES}
        for seq in valid:
            first_counts[seq[0]] += 1
        total_0 = sum(first_counts.values())
        self.first_fg = {b: first_counts[b] / total_0 for b in BASES}

        cond_counts: List[Dict[str, Dict[str, float]]] = [
            {prev: {curr: PSEUDOCOUNT for curr in BASES} for prev in BASES}
            for _ in range(self.window)
        ]

        for seq in valid:
            for i in range(1, self.window):
                prev, curr = seq[i - 1], seq[i]
                cond_counts[i][prev][curr] += 1

        self.wam_fg = []
        for i in range(self.window):
            pos_table: Dict[str, Dict[str, float]] = {}
            for prev in BASES:
                row = cond_counts[i][prev]
                total = sum(row.values())
                pos_table[prev] = {curr: row[curr] / total for curr in BASES}
            self.wam_fg.append(pos_table)

    def _validate_window(self, seq: str) -> bool:
        return bool(_validate_seq(seq, self.window))

    def score(self, seq: str) -> float:
        if self.wam_fg is None or self.first_fg is None:
            raise RuntimeError("Model not trained. Call train() first.")

        seq = seq.upper()
        if not self._validate_window(seq):
            return float("-inf")

        b0 = seq[0]
        if b0 not in BASE_IDX:
            return float("-inf")
        log_score = math.log(self.first_fg[b0] / max(self.bg[b0], 1e-9))

        for i in range(1, self.window):
            prev, curr = seq[i - 1], seq[i]
            if curr not in BASE_IDX or prev not in BASE_IDX:
                return float("-inf")
            p_fg = self.wam_fg[i][prev][curr]
            p_bg = self.bg[curr]
            log_score += math.log(p_fg / max(p_bg, 1e-9))

        return log_score

    def score_batch(self, seqs: List[str]) -> List[float]:
        return [self.score(s) for s in seqs]

    def predict(self, seq: str, threshold: float = 0.0) -> int:
        return int(self.score(seq) >= threshold)

    def predict_batch(self, seqs: List[str], threshold: float = 0.0) -> List[int]:
        return [self.predict(s, threshold) for s in seqs]

    def information_content(self) -> List[float]:
        if self.wam_fg is None or self.first_fg is None:
            raise RuntimeError("Model not trained.")
        ic = []
        ic0 = sum(self.first_fg[b] * math.log2(self.first_fg[b] / self.bg[b])
                  for b in BASES if self.first_fg[b] > 0)
        ic.append(ic0)
        for i in range(1, self.window):
            ic_i = 0.0
            for prev in BASES:
                for curr in BASES:
                    p = self.wam_fg[i][prev][curr]
                    if p > 0:
                        ic_i += 0.25 * p * math.log2(p / max(self.bg[curr], 1e-9))
            ic.append(ic_i)
        return ic

    def print_summary(self) -> None:
        if self.wam_fg is None or self.first_fg is None:
            raise RuntimeError("Model not trained. Call train() first.")
        ic = self.information_content()
        print(f"\nWAM Summary — {self.site} site  (window={self.window})")
        print("=" * 55)
        print(f"{'Pos':>4}  {'IC (bits)':>10}  {'Top conditional (prev→curr)':}")
        print("-" * 55)
        for i, ic_val in enumerate(ic):
            if i == 0:
                top_b = max(self.first_fg, key=self.first_fg.get)
                cond_str = f"marginal best: {top_b} ({self.first_fg[top_b]:.3f})"
            else:
                best_pair = max(
                    ((p, c) for p in BASES for c in BASES),
                    key=lambda pc: self.wam_fg[i][pc[0]][pc[1]]
                )
                best_val = self.wam_fg[i][best_pair[0]][best_pair[1]]
                cond_str = f"{best_pair[0]}→{best_pair[1]} ({best_val:.3f})"
            print(f"{i:>4}  {ic_val:>10.4f}  {cond_str}")
        print("=" * 55)


class DependencyWAMModel(WAMModel):
    def __init__(self, window: int, site: str = "donor", dependency_threshold: float = 6.0, max_dependency_pairs: int = 16):
        super().__init__(window, site)
        self.dependency_threshold = float(dependency_threshold)
        self.max_dependency_pairs = max(1, int(max_dependency_pairs))
        self.dependency_pairs: List[Tuple[int, int]] = []
        self.pair_log_odds: Optional[List[Dict[int, float]]] = None

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        super().train(pos_seqs, neg_seqs)
        valid_pos = [_validate_seq(s, self.window) for s in pos_seqs]
        valid_pos = [s for s in valid_pos if s]
        valid_neg = [_validate_seq(s, self.window) for s in neg_seqs]
        valid_neg = [s for s in valid_neg if s]

        chi2 = _chi2_matrix(valid_pos, self.window)
        self.dependency_pairs = _select_dependency_pairs(chi2, self.dependency_threshold, self.max_dependency_pairs)

        bg_seqs = valid_neg if valid_neg else valid_pos
        if not bg_seqs:
            self.pair_log_odds = []
            return

        self.pair_log_odds = []
        for i, j in self.dependency_pairs:
            fg_counts = {code: PSEUDOCOUNT for code in range(16)}
            bg_counts = {code: PSEUDOCOUNT for code in range(16)}
            for seq in valid_pos:
                fg_counts[_pair_index(seq[i], seq[j])] += 1.0
            for seq in bg_seqs:
                bg_counts[_pair_index(seq[i], seq[j])] += 1.0

            fg_total = sum(fg_counts.values())
            bg_total = sum(bg_counts.values())
            odds = {}
            for code in range(16):
                fg = fg_counts.get(code, PSEUDOCOUNT) / fg_total
                bg = bg_counts.get(code, PSEUDOCOUNT) / bg_total
                odds[code] = math.log(max(fg, 1e-10) / max(bg, 1e-10))
            self.pair_log_odds.append(odds)

    def score(self, seq: str) -> float:
        base_score = super().score(seq)
        if self.pair_log_odds is None:
            raise RuntimeError("Model not trained. Call train() first.")
        seq = seq.upper()
        if len(seq) != self.window:
            raise ValueError(f"Expected length {self.window}, got {len(seq)}")

        dep_score = 0.0
        for (i, j), odds in zip(self.dependency_pairs, self.pair_log_odds):
            a = seq[i]
            b = seq[j]
            if a not in BASE_IDX or b not in BASE_IDX:
                continue
            dep_score += odds.get(_pair_index(a, b), 0.0)
        return base_score + dep_score

    def print_summary(self) -> None:
        super().print_summary()
        if not self.dependency_pairs:
            print("No dependency pairs selected.")
            return
        print("Dependency pairs (chi2-selected):")
        for i, j in self.dependency_pairs[:12]:
            print(f"  ({i:>2}, {j:>2})")


def evaluate(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                sensitivity=sens, specificity=spec,
                precision=prec, F1=f1, MCC=mcc)


def roc_auc(y_true: List[int], scores: List[float], n_thresholds: int = 200) -> Tuple[List[float], List[float], float]:
    thresholds = sorted(set(scores), reverse=True)
    step = max(1, len(thresholds) // n_thresholds)
    thresholds = thresholds[::step] + [thresholds[-1]]

    fpr_list, tpr_list = [0.0], [0.0]
    for thr in thresholds:
        y_pred = [1 if s >= thr else 0 for s in scores]
        m = evaluate(y_true, y_pred)
        fpr_list.append(1.0 - m["specificity"])
        tpr_list.append(m["sensitivity"])
    fpr_list.append(1.0)
    tpr_list.append(1.0)

    auc = sum(
        (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
        for i in range(1, len(fpr_list))
    )
    return fpr_list, tpr_list, auc
