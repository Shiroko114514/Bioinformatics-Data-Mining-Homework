from typing import List, Tuple
from wam_models import WAMModel


class SpliceSiteScanner:
    DONOR_OFFSET = 3
    ACCEPTOR_OFFSET = 20

    def __init__(self, model: WAMModel, threshold: float = 0.0):
        self.model = model
        self.threshold = threshold
        self.site = model.site

    def scan(self, genome: str) -> List[Tuple[int, float]]:
        genome = genome.upper()
        w = self.model.window
        hits = []

        if self.site == "donor":
            dinuc = "GT"
            offset = self.DONOR_OFFSET
        else:
            dinuc = "AG"
            offset = self.ACCEPTOR_OFFSET

        for i in range(offset, len(genome) - w + offset + 1):
            win_start = i - offset
            win = genome[win_start: win_start + w]
            if win[offset: offset + 2] != dinuc:
                continue
            if len(win) != w:
                continue
            sc = self.model.score(win)
            if sc >= self.threshold:
                hits.append((i, sc))

        return hits
