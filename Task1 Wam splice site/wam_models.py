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
