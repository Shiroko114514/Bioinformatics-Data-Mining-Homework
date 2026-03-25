"""
WAM (Weight Array Matrix) model for eukaryotic splice site prediction
=====================================================================
References:
  Zhang & Marr (1993) – Weight Array Matrix method for splice site prediction
  Burge & Karlin (1997) – GENSCAN: prediction of complete gene structures

Splice site windows (1-indexed, cut point between exon/intron):
  Donor   (5' ss): exon[-3..-1] | intron[+1..+6]   → 9 positions
  Acceptor(3' ss): intron[-20..-1] | exon[+1..+3]   → 23 positions

Usage
-----
>>> from wam_splice_site import WAMModel, load_fasta_sequences
>>> donor_pos, donor_neg = load_fasta_sequences("donor_pos.fa", "donor_neg.fa")
>>> model = WAMModel(window=9, site="donor")
>>> model.train(donor_pos, donor_neg)
>>> score = model.score("CAGGTAAGT")
>>> preds = model.predict_batch(test_seqs, threshold=0.0)
"""

import math
import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

BASES = ("A", "C", "G", "T")
BASE_IDX = {b: i for i, b in enumerate(BASES)}

# Typical consensus positions (0-indexed) in the window
DONOR_WINDOW    = 9      # positions: exon[-3,-2,-1] | intron[+1..+6]
ACCEPTOR_WINDOW = 23     # positions: intron[-20..-1] | exon[+1..+3]

# Pseudocount to avoid log(0)
PSEUDOCOUNT = 0.5

# ─────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────

def _uniform_bg() -> Dict[str, float]:
    """Uniform background model (p = 0.25 each base)."""
    return {b: 0.25 for b in BASES}


def _empirical_bg(seqs: List[str]) -> Dict[str, float]:
    """Compute background nucleotide frequencies from a sequence set."""
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
    """Return uppercase seq if valid (correct length, only ACGT), else None."""
    seq = seq.upper().strip()
    if len(seq) != length:
        return None
    if any(ch not in BASE_IDX for ch in seq):
        return None
    return seq


# ─────────────────────────────────────────────────────────────
# WMM – Weight Matrix Model (zero-order, baseline)
# ─────────────────────────────────────────────────────────────

class WMMModel:
    """
    Position Weight Matrix (WMM / PWM) – zero-order Markov model.

    Each position is scored independently:
        log-odds(b, i) = log[ P_fg(b | pos=i) / P_bg(b) ]

    Score of a sequence s = Σᵢ log-odds(sᵢ, i)
    """

    def __init__(self, window: int, site: str = "donor"):
        self.window  = window
        self.site    = site
        self.pwm_fg: Optional[List[Dict[str, float]]] = None   # P_fg(b | i)
        self.log_odds: Optional[List[Dict[str, float]]] = None  # log(P_fg / P_bg)
        self.bg: Dict[str, float] = _uniform_bg()

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        """Estimate foreground frequencies from positive sequences."""
        valid_pos = [_validate_seq(s, self.window) for s in pos_seqs]
        valid_pos = [s for s in valid_pos if s]

        self.bg = _empirical_bg(neg_seqs)

        # Count foreground
        counts = [{b: PSEUDOCOUNT for b in BASES} for _ in range(self.window)]
        for seq in valid_pos:
            for i, ch in enumerate(seq):
                counts[i][ch] += 1

        # Normalize to frequencies
        self.pwm_fg = []
        for pos_counts in counts:
            total = sum(pos_counts.values())
            self.pwm_fg.append({b: pos_counts[b] / total for b in BASES})

        # Compute log-odds
        self.log_odds = []
        for i, freq in enumerate(self.pwm_fg):
            lo = {b: math.log(freq[b] / max(self.bg[b], 1e-9))
                  for b in BASES}
            self.log_odds.append(lo)

    def score(self, seq: str) -> float:
        """Return log-odds score for a single window sequence."""
        seq = seq.upper()
        if self.log_odds is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return sum(self.log_odds[i][ch]
                   for i, ch in enumerate(seq)
                   if ch in BASE_IDX)

    def score_batch(self, seqs: List[str]) -> List[float]:
        return [self.score(s) for s in seqs]


# ─────────────────────────────────────────────────────────────
# WAM – Weight Array Matrix (first-order Markov)
# ─────────────────────────────────────────────────────────────

class WAMModel:
    """
    Weight Array Matrix (WAM) – first-order Markov model for splice sites.

    For position i > 0, the conditional probability P(bᵢ | bᵢ₋₁) is used
    instead of the marginal P(bᵢ):

        Score(s) = log P(s₁) / P_bg(s₁)
                 + Σᵢ₌₂ᴸ  log P(sᵢ | sᵢ₋₁) / P_bg(sᵢ)

    The foreground model is represented as:
        wam_fg[i][prev_base][curr_base] = P(curr | prev, position=i)

    Attributes
    ----------
    window   : total sequence window length
    site     : "donor" or "acceptor"
    wam_fg   : conditional frequency tables (length = window)
    first_fg : marginal freq at position 0 (no conditioning)
    log_bg   : log background frequency per base
    """

    def __init__(self, window: int, site: str = "donor"):
        self.window  = window
        self.site    = site
        # wam_fg[i][prev][curr] = freq
        self.wam_fg: Optional[List[Dict[str, Dict[str, float]]]] = None
        self.first_fg: Optional[Dict[str, float]] = None
        self.bg: Dict[str, float] = _uniform_bg()

    # ── Training ──────────────────────────────────────────────

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> None:
        """
        Build the WAM from positive (true splice sites) and negative
        (background / false positive) sequences.

        Parameters
        ----------
        pos_seqs : windows centered on true splice sites
        neg_seqs : windows at random / non-splice positions (for background)
        """
        valid = [_validate_seq(s, self.window) for s in pos_seqs]
        valid = [s for s in valid if s]
        if not valid:
            raise ValueError("No valid positive sequences after filtering.")

        self.bg = _empirical_bg(neg_seqs) if neg_seqs else _uniform_bg()

        # ── Position 0: marginal counts ──
        first_counts = {b: PSEUDOCOUNT for b in BASES}
        for seq in valid:
            first_counts[seq[0]] += 1
        total_0 = sum(first_counts.values())
        self.first_fg = {b: first_counts[b] / total_0 for b in BASES}

        # ── Positions 1..L-1: conditional counts ──
        # cond_counts[i][prev][curr]
        cond_counts: List[Dict[str, Dict[str, float]]] = [
            {prev: {curr: PSEUDOCOUNT for curr in BASES} for prev in BASES}
            for _ in range(self.window)
        ]

        for seq in valid:
            for i in range(1, self.window):
                prev, curr = seq[i - 1], seq[i]
                cond_counts[i][prev][curr] += 1

        # Normalize conditional counts → probabilities
        self.wam_fg = []
        for i in range(self.window):
            pos_table: Dict[str, Dict[str, float]] = {}
            for prev in BASES:
                row = cond_counts[i][prev]
                total = sum(row.values())
                pos_table[prev] = {curr: row[curr] / total for curr in BASES}
            self.wam_fg.append(pos_table)

    # ── Scoring ───────────────────────────────────────────────

    def score(self, seq: str) -> float:
        """
        Compute the WAM log-odds score for a window sequence.

        Returns float (positive = more likely a splice site).
        """
        if self.wam_fg is None or self.first_fg is None:
            raise RuntimeError("Model not trained. Call train() first.")

        seq = seq.upper()
        if len(seq) != self.window:
            raise ValueError(f"Expected sequence of length {self.window}, got {len(seq)}.")

        # Position 0: marginal log-odds
        b0 = seq[0]
        if b0 not in BASE_IDX:
            return float("-inf")
        log_score = math.log(self.first_fg[b0] / max(self.bg[b0], 1e-9))

        # Positions 1 … L-1: conditional log-odds
        for i in range(1, self.window):
            prev, curr = seq[i - 1], seq[i]
            if curr not in BASE_IDX or prev not in BASE_IDX:
                return float("-inf")
            p_fg = self.wam_fg[i][prev][curr]
            p_bg = self.bg[curr]
            log_score += math.log(p_fg / max(p_bg, 1e-9))

        return log_score

    def score_batch(self, seqs: List[str]) -> List[float]:
        """Score a list of window sequences."""
        return [self.score(s) for s in seqs]

    def predict(self, seq: str, threshold: float = 0.0) -> int:
        """Return 1 if splice site, 0 otherwise."""
        return int(self.score(seq) >= threshold)

    def predict_batch(
        self, seqs: List[str], threshold: float = 0.0
    ) -> List[int]:
        return [self.predict(s, threshold) for s in seqs]

    # ── Inspection ────────────────────────────────────────────

    def information_content(self) -> List[float]:
        """
        Per-position information content (bits).

        IC(i) = Σ_b P_fg(b|i) × log2[ P_fg(b|i) / P_bg(b) ]

        For WAM, averages over previous base conditioning.
        """
        if self.wam_fg is None or self.first_fg is None:
            raise RuntimeError("Model not trained.")
        ic = []
        # position 0
        ic0 = sum(self.first_fg[b] * math.log2(self.first_fg[b] / self.bg[b])
                  for b in BASES if self.first_fg[b] > 0)
        ic.append(ic0)
        # positions 1..L-1: average over previous bases
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
        """Print IC per position and top dinucleotide preferences."""
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


# ─────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────

def evaluate(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Returns dict with: TP, FP, TN, FN, sensitivity, specificity,
    precision, F1, MCC (Matthews correlation coefficient).
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    sens   = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1     = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    denom  = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc    = (tp*tn - fp*fn) / denom if denom > 0 else 0.0

    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                sensitivity=sens, specificity=spec,
                precision=prec, F1=f1, MCC=mcc)


def roc_auc(
    y_true: List[int],
    scores: List[float],
    n_thresholds: int = 200,
) -> Tuple[List[float], List[float], float]:
    """
    Compute ROC curve and AUC (trapezoidal rule).

    Returns (fpr_list, tpr_list, auc).
    """
    thresholds = sorted(set(scores), reverse=True)
    # sample at most n_thresholds values
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

    # Trapezoidal AUC
    auc = sum(
        (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
        for i in range(1, len(fpr_list))
    )
    return fpr_list, tpr_list, auc


# ─────────────────────────────────────────────────────────────
# Genome scanner
# ─────────────────────────────────────────────────────────────

class SpliceSiteScanner:
    """
    Slide a WAM model over a genomic sequence to predict splice sites.

    Parameters
    ----------
    model     : trained WAMModel
    threshold : log-odds cutoff for a predicted splice site
    site      : "donor" (look for GT at canonical positions)
                or "acceptor" (look for AG)
    """

    DONOR_OFFSET    = 3   # GT starts at offset 3 in the 9-bp window
    ACCEPTOR_OFFSET = 20  # AG starts at offset 20 in the 23-bp window

    def __init__(self, model: WAMModel, threshold: float = 0.0):
        self.model     = model
        self.threshold = threshold
        self.site      = model.site

    def scan(self, genome: str) -> List[Tuple[int, float]]:
        """
        Scan a genome string and return (position, score) for predicted sites.

        `position` is the 0-indexed start of the canonical dinucleotide (GT/AG).
        """
        genome = genome.upper()
        w      = self.model.window
        hits: List[Tuple[int, float]] = []

        if self.site == "donor":
            dinuc   = "GT"
            offset  = self.DONOR_OFFSET
        else:
            dinuc   = "AG"
            offset  = self.ACCEPTOR_OFFSET

        for i in range(offset, len(genome) - w + offset + 1):
            win_start = i - offset
            win       = genome[win_start: win_start + w]
            # Quick canonical filter
            if win[offset: offset + 2] != dinuc:
                continue
            if len(win) != w:
                continue
            sc = self.model.score(win)
            if sc >= self.threshold:
                hits.append((i, sc))

        return hits


# ─────────────────────────────────────────────────────────────
# FASTA I/O
# ─────────────────────────────────────────────────────────────

def load_fasta(path: str) -> List[str]:
    """Read sequences from a FASTA file (one sequence per entry)."""
    seqs, buf = [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
            else:
                buf.append(line)
    if buf:
        seqs.append("".join(buf))
    return seqs


def load_fasta_sequences(pos_path: str, neg_path: str) -> Tuple[List[str], List[str]]:
    """Load positive and negative sample fasta files from a given paths."""
    pos_seqs = load_fasta(pos_path)
    neg_seqs = load_fasta(neg_path)
    return pos_seqs, neg_seqs


def load_txt_sequences_from_dir(dir_path: str) -> List[str]:
    """Load all sequences from *.txt files in a directory (strip whitespace)."""
    seqs: List[str] = []
    if not os.path.isdir(dir_path):
        return seqs
    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith('.txt'):
            continue
        full = os.path.join(dir_path, fname)
        if not os.path.isfile(full):
            continue
        with open(full) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    continue
                seqs.append(line)
    return seqs


# ─────────────────────────────────────────────────────────────
# Synthetic demo dataset generator
# ─────────────────────────────────────────────────────────────

def _rand_base(weights: Optional[Dict[str, float]] = None) -> str:
    bases = list(BASES)
    if weights:
        w = [weights.get(b, 0.25) for b in bases]
        total = sum(w)
        r = random.random() * total
        cum = 0.0
        for b, wi in zip(bases, w):
            cum += wi
            if r <= cum:
                return b
    return random.choice(bases)


def make_donor_positive(n: int = 500) -> List[str]:
    """
    Synthetic donor sites: window = 9  (exon[-3..-1]|intron[+1..+6])
    Consensus pattern: rAG|GTAAGT (r = purine)
    """
    seqs = []
    for _ in range(n):
        p0 = _rand_base({"A": 0.35, "G": 0.35, "C": 0.15, "T": 0.15})
        p1 = _rand_base({"A": 0.45, "G": 0.35, "C": 0.1,  "T": 0.1})
        p2 = _rand_base({"A": 0.1,  "G": 0.7,  "C": 0.1,  "T": 0.1})
        # GT is fixed (canonical donor)
        p3, p4 = "G", "T"
        p5 = _rand_base({"A": 0.65, "G": 0.15, "C": 0.1, "T": 0.1})
        p6 = _rand_base({"A": 0.7,  "G": 0.1,  "C": 0.1, "T": 0.1})
        p7 = _rand_base({"A": 0.1,  "G": 0.65, "C": 0.1, "T": 0.15})
        p8 = _rand_base({"A": 0.1,  "G": 0.1,  "C": 0.1, "T": 0.7})
        seqs.append("".join([p0, p1, p2, p3, p4, p5, p6, p7, p8]))
    return seqs


def make_donor_negative(n: int = 500) -> List[str]:
    """Random sequences that also contain GT (pseudo-negatives)."""
    seqs = []
    while len(seqs) < n:
        s = "".join(random.choices(list(BASES), k=9))
        if s[3:5] == "GT":   # same canonical position for fair comparison
            seqs.append(s)
    return seqs


# ─────────────────────────────────────────────────────────────
# Demo / main
# ─────────────────────────────────────────────────────────────

def compare_models(
    train_pos: List[str],
    train_neg: List[str],
    test_pos:  List[str],
    test_neg:  List[str],
    window:    int = DONOR_WINDOW,
    threshold: float = 0.0,
) -> None:
    """Train and compare WMM vs WAM on the same split."""

    # ── Train ──────────────────────────────────────────────────
    wmm = WMMModel(window=window, site="donor")
    wmm.train(train_pos, train_neg)

    wam = WAMModel(window=window, site="donor")
    wam.train(train_pos, train_neg)

    # ── Test set ───────────────────────────────────────────────
    test_seqs  = test_pos  + test_neg
    test_labels = [1]*len(test_pos) + [0]*len(test_neg)

    wmm_scores = wmm.score_batch(test_seqs)
    wam_scores = wam.score_batch(test_seqs)

    wmm_preds = [int(s >= threshold) for s in wmm_scores]
    wam_preds = [int(s >= threshold) for s in wam_scores]

    # ── Metrics ────────────────────────────────────────────────
    wmm_m = evaluate(test_labels, wmm_preds)
    wam_m = evaluate(test_labels, wam_preds)

    _, _, wmm_auc = roc_auc(test_labels, wmm_scores)
    _, _, wam_auc = roc_auc(test_labels, wam_scores)

    # ── Print ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  WMM vs WAM — Donor Site Prediction (5′ splice sites)")
    print("=" * 60)
    header = f"{'Metric':<18} {'WMM':>10} {'WAM':>10}"
    print(header)
    print("-" * 40)

    metrics_to_show = [
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Precision",   "precision"),
        ("F1 score",    "F1"),
        ("MCC",         "MCC"),
        ("AUC-ROC",     "__auc__"),
    ]

    for label, key in metrics_to_show:
        if key == "__auc__":
            w_val, a_val = wmm_auc, wam_auc
        else:
            w_val, a_val = wmm_m[key], wam_m[key]
        marker = " ◀" if a_val > w_val else ""
        print(f"  {label:<16} {w_val:>10.4f} {a_val:>10.4f}{marker}")

    print("-" * 40)
    print("  ◀ = WAM outperforms WMM")
    print("=" * 60)

    # WAM position-level analysis
    wam.print_summary()


def demo() -> None:
    """Run a self-contained synthetic demo, or load data from folders when available."""
    random.seed(42)

    training_dir = "/Users/shiroko/FlutterProjects/BDM Homework/Training and testing datasets/Training Set"
    testing_dir = "/Users/shiroko/FlutterProjects/BDM Homework/Training and testing datasets/Testing Set"

    train_pos = load_txt_sequences_from_dir(training_dir)
    train_neg = []  # not provided separately;训练集 txt 全部当正 samples。
    test_pos = load_txt_sequences_from_dir(testing_dir)
    test_neg = []

    if train_pos and test_pos:
        print(f"Loading data from {training_dir} and {testing_dir} ...")
    else:
        print("Training/testing directories not found or empty; generating synthetic dataset …")
        all_pos = make_donor_positive(1000)
        all_neg = make_donor_negative(1000)

        split = 800
        train_pos, test_pos = all_pos[:split], all_pos[split:]
        train_neg, test_neg = all_neg[:split], all_neg[split:]

        compare_models(train_pos, train_neg, test_pos, test_neg)
        # separate之后就直接返回，否则后续io会失败 (empty neg)
        print("Synthetic demo finished.")
        return

    # 默认训练+测试路径机制：如果训练/测试目录里有序列，则用它们；
    #   训练集为正样本，测试集为负样本（你可以根据需求改成分文件或正负分开）
    if not test_neg:
        print("Warning: test_neg is empty; using test_pos as neg for evaluation fallback.")
        test_neg = test_pos[:len(test_pos)//2]
        test_pos = test_pos[len(test_pos)//2:]

    if not train_neg:
        print("Warning: train_neg is empty; using random negatives from synthetic generator fallback.")
        train_neg = make_donor_negative(len(train_pos))

    compare_models(train_pos, train_neg, test_pos, test_neg)

    # ── Genome scan demo ───────────────────────────────────────
    print("\n── Genome scan demo ──────────────────────────────")
    wam = WAMModel(window=DONOR_WINDOW, site="donor")
    wam.train(train_pos, train_neg)
    scanner = SpliceSiteScanner(wam, threshold=1.0)

    # Synthetic genome fragment with two planted donor sites
    genome = ("ATCGATCGATCG"
              "CAGGTAAGTATCG"    # planted donor 1 at offset 12
              "GCATCGATCGATCG"
              "AAGGTAAGTGCTA"    # planted donor 2 at offset 39
              "TTTGCATCGATCG")

    hits = scanner.scan(genome)
    print(f"Genome length: {len(genome)} bp")
    print(f"Predicted donor sites (threshold=1.0):")
    for pos, sc in hits:
        context = genome[max(0, pos-2): pos+5]
        print(f"  position {pos:>4}  score={sc:+.3f}  context: …{context}…")
    if not hits:
        print("  (none above threshold)")


if __name__ == "__main__":
    demo()