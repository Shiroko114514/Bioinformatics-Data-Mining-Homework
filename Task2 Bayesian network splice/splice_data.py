import re
import random
from pathlib import Path
from typing import List, Tuple, Set
from splice_utils import BASES


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
    p = Path(dir_path)
    for txt_file in sorted(p.glob('*.TXT')) + sorted(p.glob('*.txt')):
        if txt_file.name.endswith('.txt') and Path(str(txt_file).replace('.txt', '.TXT')).exists():
            continue
        seq, ranges = parse_genbank_file(str(txt_file))
        if seq and ranges:
            all_sites.extend(extract_splice_sites_from_ranges(seq, ranges,
                                                            site_type=site_type,
                                                            window=window))
    return all_sites


def load_sequences_from_dir(dir_path: str) -> List[str]:
    result = []
    p = Path(dir_path)
    for txt_file in sorted(p.glob('*.TXT')) + sorted(p.glob('*.txt')):
        if txt_file.name.endswith('.txt') and Path(str(txt_file).replace('.txt', '.TXT')).exists():
            continue
        seq, _ = parse_genbank_file(str(txt_file))
        if seq:
            result.append(seq)
    return result


from typing import Optional


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
