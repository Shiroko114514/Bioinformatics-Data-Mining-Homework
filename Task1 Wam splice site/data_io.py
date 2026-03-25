import os
from typing import List, Tuple
from wam_models import DONOR_WINDOW, ACCEPTOR_WINDOW, BASE_IDX


def load_fasta(path: str) -> List[str]:
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


def load_fasta_sequences(pos_path: str, neg_path: str):
    return load_fasta(pos_path), load_fasta(neg_path)


def _clean_dna_text(raw: str) -> str:
    return ''.join(ch for ch in raw.upper() if ch in BASE_IDX)


def load_txt_sequences_from_dir(
    dir_path: str,
    window: int = DONOR_WINDOW,
    site: str = "donor",
    max_windows: int = 5000,
) -> Tuple[List[str], List[str]]:
    positives, negatives = [], []
    if not os.path.isdir(dir_path):
        return positives, negatives

    offset = 3 if site == "donor" else 20
    canonical = "GT" if site == "donor" else "AG"

    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith('.txt'):
            continue
        path = os.path.join(dir_path, fname)
        if not os.path.isfile(path):
            continue

        raw = []
        with open(path) as fh:
            for line in fh:
                raw.append(line.strip())
        seq = _clean_dna_text(''.join(raw))

        for i in range(0, len(seq) - window + 1):
            win = seq[i:i + window]
            if win[offset:offset + 2] == canonical:
                positives.append(win)
            else:
                negatives.append(win)

        if len(positives) >= max_windows and len(negatives) >= max_windows:
            break

    return positives[:max_windows], negatives[:max_windows]


def _rand_base(weights=None):
    import random
    bases = list(BASE_IDX.keys())
    if weights:
        w = [weights.get(b, 0.25) for b in bases]
        total = sum(w)
        r = random.random() * total
        c = 0.0
        for b, wi in zip(bases, w):
            c += wi
            if r <= c:
                return b
    return random.choice(bases)


def make_donor_positive(n=500):
    seqs = []
    for _ in range(n):
        p0 = _rand_base({"A": 0.35, "G": 0.35, "C": 0.15, "T": 0.15})
        p1 = _rand_base({"A": 0.45, "G": 0.35, "C": 0.1, "T": 0.1})
        p2 = _rand_base({"A": 0.1, "G": 0.7, "C": 0.1, "T": 0.1})
        p3, p4 = "G", "T"
        p5 = _rand_base({"A": 0.65, "G": 0.15, "C": 0.1, "T": 0.1})
        p6 = _rand_base({"A": 0.7, "G": 0.1, "C": 0.1, "T": 0.1})
        p7 = _rand_base({"A": 0.1, "G": 0.65, "C": 0.1, "T": 0.15})
        p8 = _rand_base({"A": 0.1, "G": 0.1, "C": 0.1, "T": 0.7})
        seqs.append(''.join([p0, p1, p2, p3, p4, p5, p6, p7, p8]))
    return seqs


def make_donor_negative(n=500):
    import random
    seqs = []
    while len(seqs) < n:
        s = ''.join(random.choices(list(BASE_IDX), k=9))
        if s[3:5] == 'GT':
            seqs.append(s)
    return seqs
