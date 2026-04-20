from typing import Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

from splice_features import FeatureExtractor
from splice_utils import ACCEPTOR_AG_POS, DONOR_GT_POS, DONOR_WINDOW, validate_seqs


class SVMSpliceSite:
    def __init__(
        self,
        window: int = DONOR_WINDOW,
        site: str = "donor",
        kernel: Literal["linear", "rbf", "poly", "sigmoid", "precomputed"] = "rbf",
        feature_set: Union[str, List[str]] = "combined",
        dependency_threshold: float = 6.0,
        max_dependency_pairs: int = 16,
        ebn_max_parents: int = 2,
        C: float = 1.0,
        gamma: Union[float, Literal["scale", "auto"]] = "scale",
        degree: int = 3,
        probability: bool = True,
    ) -> None:
        self.window = window
        self.site = site
        self.kernel = kernel
        self.feature_set = feature_set
        self.dependency_threshold = dependency_threshold
        self.max_dependency_pairs = max_dependency_pairs
        self.ebn_max_parents = ebn_max_parents
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.probability = probability

        self.extractor: Optional[FeatureExtractor] = None
        self.pipeline: Optional[Pipeline] = None
        self._train_pos: Optional[List[str]] = None
        self._train_neg: Optional[List[str]] = None

    def train(self, pos_seqs: List[str], neg_seqs: List[str]) -> "SVMSpliceSite":
        pos = validate_seqs(pos_seqs, self.window)
        neg = validate_seqs(neg_seqs, self.window)
        if not pos:
            raise ValueError("No valid positive sequences after filtering.")
        self._train_pos = pos
        self._train_neg = neg

        self.extractor = FeatureExtractor(
            self.window,
            self.feature_set,
            dependency_threshold=self.dependency_threshold,
            max_dependency_pairs=self.max_dependency_pairs,
            ebn_max_parents=self.ebn_max_parents,
        )
        self.extractor.fit(pos, neg)

        X_pos = self.extractor.transform(pos)
        X_neg = self.extractor.transform(neg)
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(pos) + [-1] * len(neg), dtype=int)

        svc = SVC(
            kernel=cast(Literal["linear", "rbf", "poly", "sigmoid", "precomputed"], self.kernel),
            C=self.C,
            gamma=cast(Union[float, Literal["scale", "auto"]], self.gamma),
            degree=self.degree,
            probability=self.probability,
            class_weight="balanced",
            random_state=42,
        )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", svc),
        ])
        self.pipeline.fit(X, y)
        return self

    def _extract(self, seqs: Union[str, List[str]]) -> np.ndarray:
        if self.extractor is None:
            raise RuntimeError("Call train() first.")
        if isinstance(seqs, str):
            seqs = [seqs]
        seqs = [s.upper() for s in seqs]
        return self.extractor.transform(seqs)

    def decision_score(self, seq: str) -> float:
        if self.pipeline is None:
            raise RuntimeError("Call train() first.")
        X = self._extract(seq)
        return float(self.pipeline.decision_function(X)[0])

    def decision_score_batch(self, seqs: List[str]) -> List[float]:
        if self.pipeline is None:
            raise RuntimeError("Call train() first.")
        X = self._extract(seqs)
        return self.pipeline.decision_function(X).tolist()

    def predict(self, seq: str, threshold: float = 0.0) -> int:
        return 1 if self.decision_score(seq) >= threshold else -1

    def predict_batch(self, seqs: List[str], threshold: float = 0.0) -> List[int]:
        return [self.predict(s, threshold) for s in seqs]

    def predict_proba(self, seq: str) -> float:
        if not self.probability:
            raise RuntimeError("Set probability=True at construction.")
        if self.pipeline is None:
            raise RuntimeError("Call train() first.")
        X = self._extract(seq)
        return float(self.pipeline.predict_proba(X)[0, 1])

    def cross_validate(self, pos_seqs: List[str], neg_seqs: List[str], n_folds: int = 5) -> Dict[str, float]:
        pos = validate_seqs(pos_seqs, self.window)
        neg = validate_seqs(neg_seqs, self.window)

        if len(pos) < n_folds or len(neg) < n_folds:
            raise ValueError("Not enough samples for the requested number of folds.")

        all_seqs = pos + neg
        y = np.array([1] * len(pos) + [-1] * len(neg), dtype=int)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        auc_scores: List[float] = []
        mcc_scores: List[float] = []
        f1_scores: List[float] = []

        for train_idx, test_idx in cv.split(all_seqs, y):
            train_seqs = [all_seqs[i] for i in train_idx]
            test_seqs = [all_seqs[i] for i in test_idx]
            train_labels = y[train_idx]
            test_labels = y[test_idx]

            train_pos_fold = [seq for seq, label in zip(train_seqs, train_labels) if label == 1]
            train_neg_fold = [seq for seq, label in zip(train_seqs, train_labels) if label == -1]
            extractor = FeatureExtractor(
                self.window,
                self.feature_set,
                dependency_threshold=self.dependency_threshold,
                max_dependency_pairs=self.max_dependency_pairs,
                ebn_max_parents=self.ebn_max_parents,
            ).fit(train_pos_fold, train_neg_fold)

            X_train = extractor.transform(train_seqs)
            X_test = extractor.transform(test_seqs)

            svc = SVC(
                kernel=cast(Literal["linear", "rbf", "poly", "sigmoid", "precomputed"], self.kernel),
                C=self.C,
                gamma=cast(Union[float, Literal["scale", "auto"]], self.gamma),
                degree=self.degree,
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
            pipe = Pipeline([("scaler", StandardScaler()), ("svm", svc)])
            pipe.fit(X_train, train_labels)

            scores = pipe.decision_function(X_test)
            preds = np.where(scores >= 0.0, 1, -1)

            auc_scores.append(float(roc_auc_score(test_labels, scores)))
            mcc_scores.append(float(matthews_corrcoef(test_labels, preds)))
            f1_scores.append(float(f1_score(test_labels, preds, pos_label=1, zero_division=0)))

        return {
            "auc_mean": float(np.mean(auc_scores)),
            "auc_std": float(np.std(auc_scores)),
            "mcc_mean": float(np.mean(mcc_scores)),
            "mcc_std": float(np.std(mcc_scores)),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
        }

    def top_features(self, n: int = 15) -> List[Tuple[str, float]]:
        if self.pipeline is None:
            raise RuntimeError("Call train() first.")
        if self.extractor is None:
            raise RuntimeError("Call train() first.")
        if self.kernel != "linear":
            raise ValueError("Feature weights are only interpretable for linear kernel.")
        svm = self.pipeline.named_steps["svm"]
        weights = svm.coef_[0]
        names = self.extractor.feature_names()
        ranked = sorted(zip(names, weights.tolist()), key=lambda x: abs(x[1]), reverse=True)
        return ranked[:n]

    def scan(self, genome: str, threshold: float = 0.0) -> List[Tuple[int, float]]:
        genome = genome.upper()
        w = self.window
        dinuc = "GT" if self.site == "donor" else "AG"
        off = DONOR_GT_POS if self.site == "donor" else ACCEPTOR_AG_POS
        hits: List[Tuple[int, float]] = []
        for i in range(off, len(genome) - w + off + 1):
            win = genome[i - off : i - off + w]
            if len(win) != w:
                continue
            if win[off : off + 2] != dinuc:
                continue
            sc = self.decision_score(win)
            if sc >= threshold:
                hits.append((i, sc))
        return hits

    def print_summary(self) -> None:
        if self.pipeline is None:
            print("Not trained.")
            return
        if self.extractor is None:
            print("Not trained.")
            return

        svm = self.pipeline.named_steps["svm"]
        n_sv = svm.n_support_.sum()
        dim = self.extractor.feature_dim

        print(f"\nSVM Splice Site Predictor - {self.site} site")
        print("=" * 52)
        print(f"  Window      : {self.window} bp")
        print(f"  Kernel      : {self.kernel}")
        print(f"  C           : {self.C}")
        print(f"  Gamma       : {self.gamma}")
        print(f"  Features    : {', '.join(self.extractor.features)}")
        print(f"  Feature dim : {dim}")
        print(f"  Support vecs: {n_sv}")
        if self.kernel == "linear":
            svm_weights = svm.coef_[0]
            print(f"  ||w||2      : {float(np.linalg.norm(svm_weights)):.4f}")
            print(f"  Margin width: {2.0 / float(np.linalg.norm(svm_weights)):.4f}")
        print("=" * 52)
