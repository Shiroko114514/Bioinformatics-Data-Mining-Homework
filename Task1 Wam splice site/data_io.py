import os
import random
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union
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


def _iter_txt_files(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    files = sorted(p.glob("*.TXT")) + sorted(p.glob("*.txt"))
    unique_files: List[Path] = []
    seen_names = set()
    for txt_file in files:
        key = txt_file.name.upper()
        if key in seen_names:
            continue
        seen_names.add(key)
        unique_files.append(txt_file)
    return unique_files


def parse_genbank_file(filepath: str) -> Tuple[str, List[Tuple[int, int]]]:
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as exc:
        print(f"Error reading {filepath}: {exc}")
        return "", []

    lines = content.split("\n")
    if len(lines) < 3:
        return "", []

    ranges: List[Tuple[int, int]] = []
    is_fasta = lines[0].startswith(">")

    if is_fasta:
        if len(lines) > 1 and "(" in lines[1]:
            match = re.search(r"\(([^)]+)\)", lines[1])
            if match:
                for range_part in match.group(1).split(","):
                    part = range_part.strip()
                    if ".." in part:
                        try:
                            start, end = part.split("..")
                            ranges.append((int(start), int(end)))
                        except ValueError:
                            continue
        dna_start = 2
    else:
        for line in lines[:10]:
            if "CDS" in line and "join" in line:
                match = re.search(r"join\(\s*(.+?)\s*\)", line)
                if match:
                    for range_part in match.group(1).split(","):
                        part = range_part.strip()
                        if ".." in part:
                            try:
                                start, end = part.split("..")
                                ranges.append((int(start), int(end)))
                            except ValueError:
                                continue
                break
        dna_start = 2

    dna_lines = []
    for line in lines[dna_start:]:
        s = line.strip()
        if not s or s.startswith(">") or any(x in s.upper() for x in ["LOCUS", "CDS", "//"]):
            continue
        if all(c.lower() in "acgtn" or c.isspace() for c in s):
            dna_lines.append(s)

    seq = "".join(dna_lines).upper()
    seq = "".join(c for c in seq if c in BASE_IDX)
    return seq, ranges


def _canonical_signature(site_type: str) -> Tuple[str, int]:
    if site_type == "donor":
        return "GT", 3
    if site_type == "acceptor":
        return "AG", 20
    raise ValueError(f"Unknown site type: {site_type}")


def _collect_canonical_windows(seq: str, window: int, site_type: str) -> List[str]:
    dinuc, dinuc_offset = _canonical_signature(site_type)
    out: List[str] = []
    if len(seq) < window:
        return out
    for i in range(0, len(seq) - window + 1):
        win = seq[i:i + window]
        if win[dinuc_offset:dinuc_offset + 2] == dinuc:
            out.append(win)
    return out


def extract_splice_sites_from_ranges(
    seq: str,
    ranges: List[Tuple[int, int]],
    site_type: str = "donor",
    window: int = DONOR_WINDOW,
) -> List[str]:
    sites = []
    dinuc, dinuc_offset = _canonical_signature(site_type)

    for start, end in ranges:
        start_0 = start - 1
        end_0 = end - 1
        if site_type == "donor":
            site_pos = end_0 + 1
        else:
            site_pos = start_0 - 2

        if site_pos < 0 or site_pos + 1 >= len(seq):
            continue
        if seq[site_pos:site_pos + 2] != dinuc:
            continue

        win_start = max(0, site_pos - dinuc_offset)
        win_end = win_start + window
        if win_end > len(seq):
            continue
        window_seq = seq[win_start:win_end]
        if window_seq[dinuc_offset:dinuc_offset + 2] == dinuc:
            sites.append(window_seq)

    return sites


def _unique_keep_order(seqs: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seqs:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def load_positive_sites_from_dir(
    dir_path: str,
    site_type: str = "donor",
    window: int = DONOR_WINDOW,
) -> List[str]:
    all_sites = []
    for txt_file in _iter_txt_files(dir_path):
        seq, ranges = parse_genbank_file(str(txt_file))
        if seq and ranges:
            all_sites.extend(extract_splice_sites_from_ranges(seq, ranges, site_type=site_type, window=window))
    return _unique_keep_order(all_sites)


def load_hard_negative_sites_from_dir(
    dir_path: str,
    site_type: str = "donor",
    window: int = DONOR_WINDOW,
    exclude_sites: Optional[Set[str]] = None,
) -> List[str]:
    exclude = exclude_sites if exclude_sites is not None else set()
    all_sites: List[str] = []
    for txt_file in _iter_txt_files(dir_path):
        seq, _ = parse_genbank_file(str(txt_file))
        if not seq:
            continue
        for win in _collect_canonical_windows(seq, window=window, site_type=site_type):
            if win not in exclude:
                all_sites.append(win)
    return _unique_keep_order(all_sites)


def load_strict_dataset_split(
    base_path: Optional[Union[str, Path]] = None,
    site: str = "donor",
    window: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    if window is None:
        window = DONOR_WINDOW if site == "donor" else ACCEPTOR_WINDOW
    base = (
        Path(base_path)
        if base_path is not None
        else Path(__file__).resolve().parent.parent / "Training and testing datasets"
    )
    train_dir = base / "Training Set"
    test_dir = base / "Testing Set"
    if train_dir.resolve() == test_dir.resolve():
        raise RuntimeError("Training Set and Testing Set must be different directories.")

    train_pos = load_positive_sites_from_dir(str(train_dir), site_type=site, window=window)
    test_pos_all = load_positive_sites_from_dir(str(test_dir), site_type=site, window=window)

    train_pos_set = set(train_pos)
    test_pos = [w for w in test_pos_all if w not in train_pos_set]

    train_neg_pool = load_hard_negative_sites_from_dir(
        str(train_dir),
        site_type=site,
        window=window,
        exclude_sites=train_pos_set,
    )
    train_neg_set = set(train_neg_pool)
    test_neg_pool = load_hard_negative_sites_from_dir(
        str(test_dir),
        site_type=site,
        window=window,
        exclude_sites=set(test_pos) | train_pos_set | train_neg_set,
    )

    rng = random.Random(seed)
    rng.shuffle(train_pos)
    rng.shuffle(test_pos)
    rng.shuffle(train_neg_pool)
    rng.shuffle(test_neg_pool)

    n_train = min(len(train_pos), len(train_neg_pool))
    n_test = min(len(test_pos), len(test_neg_pool))
    if n_train == 0 or n_test == 0:
        raise RuntimeError("Strict split produced an empty train/test set. Check data parsing and constraints.")

    return (
        train_pos[:n_train],
        train_neg_pool[:n_train],
        test_pos[:n_test],
        test_neg_pool[:n_test],
    )


def load_txt_sequences_from_dir(
    dir_path: str,
    window: int = DONOR_WINDOW,
    site: str = "donor",
    max_windows: Optional[int] = None,
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

        if max_windows is not None and len(positives) >= max_windows and len(negatives) >= max_windows:
            break

    if max_windows is None:
        return positives, negatives
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


def make_acceptor_positive(n=500):
    seqs = []
    for _ in range(n):
        p0 = _rand_base({"A": 0.35, "G": 0.35, "C": 0.15, "T": 0.15})
        p1 = _rand_base({"A": 0.45, "G": 0.35, "C": 0.1, "T": 0.1})
        p2 = _rand_base({"A": 0.7, "G": 0.1, "C": 0.1, "T": 0.1})
        p3 = _rand_base({"A": 0.1, "G": 0.65, "C": 0.1, "T": 0.15})
        p4 = _rand_base({"A": 0.7, "G": 0.1, "C": 0.1, "T": 0.1})
        p5 = _rand_base({"A": 0.65, "G": 0.15, "C": 0.1, "T": 0.1})
        p6 = _rand_base({"A": 0.1, "G": 0.1, "C": 0.1, "T": 0.7})
        p7 = _rand_base({"A": 0.1, "G": 0.65, "C": 0.1, "T": 0.15})
        p8 = _rand_base({"A": 0.1, "G": 0.1, "C": 0.1, "T": 0.7})
        p9 = _rand_base({"A": 0.1, "G": 0.7, "C": 0.1, "T": 0.1})
        p10 = _rand_base({"A": 0.35, "G": 0.25, "C": 0.2, "T": 0.2})
        p11 = _rand_base({"A": 0.25, "G": 0.25, "C": 0.25, "T": 0.25})
        p12 = _rand_base({"A": 0.25, "G": 0.25, "C": 0.25, "T": 0.25})
        p13 = _rand_base({"A": 0.2, "G": 0.2, "C": 0.3, "T": 0.3})
        p14 = _rand_base({"A": 0.2, "G": 0.2, "C": 0.3, "T": 0.3})
        p15 = _rand_base({"A": 0.2, "G": 0.25, "C": 0.25, "T": 0.3})
        p16 = _rand_base({"A": 0.15, "G": 0.2, "C": 0.25, "T": 0.4})
        p17 = _rand_base({"A": 0.2, "G": 0.2, "C": 0.25, "T": 0.35})
        p18 = _rand_base({"A": 0.2, "G": 0.2, "C": 0.3, "T": 0.3})
        p19 = _rand_base({"A": 0.25, "G": 0.25, "C": 0.25, "T": 0.25})
        p20, p21 = "A", "G"
        p22 = _rand_base({"A": 0.15, "G": 0.4, "C": 0.2, "T": 0.25})
        seqs.append(''.join([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22]))
    return seqs


def make_acceptor_negative(n=500):
    import random
    seqs = []
    while len(seqs) < n:
        s = ''.join(random.choices(list(BASE_IDX), k=23))
        if s[20:22] == 'AG':
            seqs.append(s)
    return seqs
