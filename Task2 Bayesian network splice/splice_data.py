import re
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union
from splice_utils import ACCEPTOR_WINDOW, DONOR_WINDOW, BASES


def _iter_txt_files(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    files = sorted(p.glob('*.TXT')) + sorted(p.glob('*.txt'))
    unique_files: List[Path] = []
    seen_names = set()
    for txt_file in files:
        key = txt_file.name.upper()
        if key in seen_names:
            continue
        seen_names.add(key)
        unique_files.append(txt_file)
    return unique_files


def _canonical_signature(site_type: str) -> Tuple[str, int]:
    if site_type == 'donor':
        return 'GT', 3
    if site_type == 'acceptor':
        return 'AG', 20
    raise ValueError(f'Unknown site type: {site_type}')


def _unique_keep_order(seqs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seqs:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


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


def parse_genbank_file(filepath: str) -> Tuple[str, List[Tuple[int, int]]]:
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return '', []

    lines = content.split('\n')
    if len(lines) < 3:
        return '', []

    ranges = []
    is_fasta = lines[0].startswith('>')

    if is_fasta:
        if len(lines) > 1 and '(' in lines[1]:
            header_line = lines[1]
            match = re.search(r'\(([^)]+)\)', header_line)
            if match:
                for range_part in match.group(1).split(','):
                    part = range_part.strip()
                    if '..' in part:
                        try:
                            start, end = part.split('..')
                            ranges.append((int(start), int(end)))
                        except ValueError:
                            continue
        dna_start = 2
    else:
        for i, line in enumerate(lines[:10]):
            if 'CDS' in line and 'join' in line:
                match = re.search(r'join\(\s*(.+?)\s*\)', line)
                if match:
                    for range_part in match.group(1).split(','):
                        part = range_part.strip()
                        if '..' in part:
                            try:
                                start, end = part.split('..')
                                ranges.append((int(start), int(end)))
                            except ValueError:
                                continue
                break
        dna_start = 2

    dna_lines = []
    for line in lines[dna_start:]:
        s = line.strip()
        if not s or s.startswith('>') or any(x in s.upper() for x in ['LOCUS', 'CDS', '//']):
            continue
        if all(c.lower() in 'acgtN' or c.isspace() for c in s):
            dna_lines.append(s)

    seq = ''.join(dna_lines).upper()
    seq = ''.join(c for c in seq if c in BASES)
    return seq, ranges


def extract_splice_sites_from_ranges(seq: str, ranges: List[Tuple[int, int]],
                                     site_type: str = 'donor', window: int = 9) -> List[str]:
    sites = []
    dinuc = 'GT' if site_type == 'donor' else 'AG'
    dinuc_offset = 3 if site_type == 'donor' else 20

    for start, end in ranges:
        start_0 = start - 1
        end_0 = end - 1
        if site_type == 'donor':
            gt_pos = end_0 + 1
            if gt_pos + 1 < len(seq) and seq[gt_pos:gt_pos + 2] == 'GT':
                win_start = max(0, gt_pos - dinuc_offset)
                win_end = win_start + window
                if win_end <= len(seq):
                    window_seq = seq[win_start:win_end]
                    if window_seq[dinuc_offset:dinuc_offset + 2] == dinuc:
                        sites.append(window_seq)
        else:
            ag_pos = start_0 - 2
            if ag_pos >= 0 and ag_pos + 1 < len(seq) and seq[ag_pos:ag_pos + 2] == 'AG':
                win_start = max(0, ag_pos - dinuc_offset)
                win_end = win_start + window
                if win_end <= len(seq):
                    window_seq = seq[win_start:win_end]
                    if window_seq[dinuc_offset:dinuc_offset + 2] == dinuc:
                        sites.append(window_seq)
    return sites


def load_positive_sites_from_dir(dir_path: str, site_type: str = 'donor', window: int = 9) -> List[str]:
    all_sites = []
    for txt_file in _iter_txt_files(dir_path):
        seq, ranges = parse_genbank_file(str(txt_file))
        if seq and ranges:
            all_sites.extend(extract_splice_sites_from_ranges(seq, ranges,
                                                            site_type=site_type,
                                                            window=window))
    return _unique_keep_order(all_sites)


def load_sequences_from_dir(dir_path: str) -> List[str]:
    result = []
    for txt_file in _iter_txt_files(dir_path):
        seq, _ = parse_genbank_file(str(txt_file))
        if seq:
            result.append(seq)
    return result


def load_hard_negative_sites_from_dir(
    dir_path: str,
    site_type: str = 'donor',
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
    site: str = 'donor',
    window: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    if window is None:
        window = DONOR_WINDOW if site == 'donor' else ACCEPTOR_WINDOW

    base = (
        Path(base_path)
        if base_path is not None
        else Path(__file__).resolve().parent.parent / 'Training and testing datasets'
    )
    train_dir = base / 'Training Set'
    test_dir = base / 'Testing Set'
    if train_dir.resolve() == test_dir.resolve():
        raise RuntimeError('Training Set and Testing Set must be different directories.')

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
        raise RuntimeError('Strict split produced an empty train/test set. Check data parsing and constraints.')

    return (
        train_pos[:n_train],
        train_neg_pool[:n_train],
        test_pos[:n_test],
        test_neg_pool[:n_test],
    )


def generate_negative_samples(all_sequences: List[str], n_neg: int, window: int = 9,
                               exclude_sites: Optional[Set[str]] = None) -> List[str]:
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
        if len(w) == window and all(c in BASES for c in w) and w not in exclude_sites:
            negs.append(w)
    return negs[:n_neg]
