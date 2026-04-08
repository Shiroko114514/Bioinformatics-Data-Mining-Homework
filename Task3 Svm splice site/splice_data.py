import re
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from splice_utils import (
    ACCEPTOR_AG_POS,
    ACCEPTOR_WINDOW,
    BASE_IDX,
    DONOR_GT_POS,
    DONOR_WINDOW,
    rand_base,
)


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


def extract_splice_sites_from_ranges(
    seq: str,
    ranges: List[Tuple[int, int]],
    site_type: str = "donor",
    window: int = DONOR_WINDOW,
) -> List[str]:
    sites = []
    dinuc = "GT" if site_type == "donor" else "AG"
    dinuc_offset = DONOR_GT_POS if site_type == "donor" else ACCEPTOR_AG_POS

    for start, end in ranges:
        start_0 = start - 1
        end_0 = end - 1
        if site_type == "donor":
            gt_pos = end_0 + 1
            if gt_pos + 1 < len(seq) and seq[gt_pos:gt_pos + 2] == "GT":
                win_start = max(0, gt_pos - dinuc_offset)
                win_end = win_start + window
                if win_end <= len(seq):
                    window_seq = seq[win_start:win_end]
                    if window_seq[dinuc_offset:dinuc_offset + 2] == dinuc:
                        sites.append(window_seq)
        else:
            ag_pos = start_0 - 2
            if ag_pos >= 0 and ag_pos + 1 < len(seq) and seq[ag_pos:ag_pos + 2] == "AG":
                win_start = max(0, ag_pos - dinuc_offset)
                win_end = win_start + window
                if win_end <= len(seq):
                    window_seq = seq[win_start:win_end]
                    if window_seq[dinuc_offset:dinuc_offset + 2] == dinuc:
                        sites.append(window_seq)
    return sites


def load_positive_sites_from_dir(
    dir_path: str,
    site_type: str = "donor",
    window: int = DONOR_WINDOW,
) -> List[str]:
    all_sites = []
    p = Path(dir_path)
    for txt_file in sorted(p.glob("*.TXT")) + sorted(p.glob("*.txt")):
        if txt_file.name.endswith(".txt") and Path(str(txt_file).replace(".txt", ".TXT")).exists():
            continue
        seq, ranges = parse_genbank_file(str(txt_file))
        if seq and ranges:
            all_sites.extend(
                extract_splice_sites_from_ranges(seq, ranges, site_type=site_type, window=window)
            )
    return all_sites


def load_sequences_from_dir(dir_path: str) -> List[str]:
    result = []
    p = Path(dir_path)
    for txt_file in sorted(p.glob("*.TXT")) + sorted(p.glob("*.txt")):
        if txt_file.name.endswith(".txt") and Path(str(txt_file).replace(".txt", ".TXT")).exists():
            continue
        seq, _ = parse_genbank_file(str(txt_file))
        if seq:
            result.append(seq)
    return result


def generate_negative_samples(
    all_sequences: List[str],
    n_neg: int,
    window: int = DONOR_WINDOW,
    exclude_sites: Optional[Set[str]] = None,
) -> List[str]:
    if exclude_sites is None:
        exclude_sites = set()
    negs = []
    attempts = n_neg * 100
    for _ in range(attempts):
        if len(negs) >= n_neg:
            break
        seq = random.choice(all_sequences)
        if len(seq) < window:
            continue
        start = random.randint(0, len(seq) - window)
        w = seq[start:start + window]
        if len(w) == window and all(c in BASE_IDX for c in w) and w not in exclude_sites:
            negs.append(w)
    return negs[:n_neg]


def load_real_dataset_split(
    base_path: Optional[Union[str, Path]] = None,
    site: str = "donor",
    window: Optional[int] = None,
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

    train_pos = load_positive_sites_from_dir(str(train_dir), site_type=site, window=window)
    test_pos = load_positive_sites_from_dir(str(test_dir), site_type=site, window=window)
    train_seqs = load_sequences_from_dir(str(train_dir))
    test_seqs = load_sequences_from_dir(str(test_dir))

    train_neg = generate_negative_samples(train_seqs, len(train_pos), window=window, exclude_sites=set(train_pos))
    test_neg = generate_negative_samples(test_seqs, len(test_pos), window=window, exclude_sites=set(test_pos))
    return train_pos, train_neg, test_pos, test_neg


def make_donor_positive(n: int = 500) -> List[str]:
    seqs = []
    for _ in range(n):
        p0 = rand_base({"A": 0.35, "G": 0.35, "C": 0.15, "T": 0.15})
        p1 = rand_base({"A": 0.45, "G": 0.35, "C": 0.10, "T": 0.10})
        p2 = (
            "G"
            if p0 in ("A", "G") and random.random() < 0.70
            else rand_base({"A": 0.20, "G": 0.50, "C": 0.15, "T": 0.15})
        )
        p3, p4 = "G", "T"
        p5 = rand_base({"A": 0.65, "G": 0.15, "C": 0.10, "T": 0.10})
        p6 = rand_base({"A": 0.70, "G": 0.10, "C": 0.10, "T": 0.10})
        p7 = rand_base({"A": 0.10, "G": 0.65, "C": 0.10, "T": 0.15})
        p8 = (
            "T"
            if p5 == "A" and random.random() < 0.75
            else rand_base({"A": 0.10, "G": 0.10, "C": 0.10, "T": 0.70})
        )
        seqs.append("".join([p0, p1, p2, p3, p4, p5, p6, p7, p8]))
    return seqs


def make_donor_negative(n: int = 500) -> List[str]:
    seqs = []
    while len(seqs) < n:
        s = list("".join(random.choices(list(BASE_IDX), k=9)))
        s[3], s[4] = "G", "T"
        seqs.append("".join(s))
    return seqs
