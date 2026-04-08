import math
import random
from typing import Dict, List

BASES = ("A", "C", "G", "T")
BASE_IDX = {b: i for i, b in enumerate(BASES)}
PSEUDOCOUNT = 0.5

DONOR_WINDOW = 9
ACCEPTOR_WINDOW = 23
DONOR_GT_POS = 3
ACCEPTOR_AG_POS = 20


def validate_seqs(seqs: List[str], window: int) -> List[str]:
    out: List[str] = []
    for s in seqs:
        s = s.upper().strip()
        if len(s) == window and all(c in BASE_IDX for c in s):
            out.append(s)
    return out


def rand_base(weights: Dict[str, float]) -> str:
    r, cum = random.random(), 0.0
    for b, w in weights.items():
        cum += w
        if r < cum:
            return b
    return "A"


def log_odds(p_fg: float, p_bg: float) -> float:
    return math.log(max(p_fg, 1e-10) / max(p_bg, 1e-10))
