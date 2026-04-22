import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, roc_auc_score

from splice_features import FeatureExtractor
from splice_model import SVMSpliceSite
from splice_utils import BASE_IDX, BASES, DONOR_WINDOW, PSEUDOCOUNT, validate_seqs
from roc_plot import plot_roc_curves


def evaluate_full(
    y_true: List[int],
    y_pred: List[int],
    scores: Optional[List[float]] = None,
) -> Dict[str, float]:
    yt = np.array(y_true)
    yp = np.array(y_pred)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[-1, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(yt, yp, pos_label=1, zero_division=0)
    f1 = f1_score(yt, yp, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(yt, yp)
    auc = roc_auc_score(yt, scores) if scores is not None else float("nan")
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "F1": float(f1),
        "MCC": float(mcc),
        "AUC": float(auc),
    }


def ablation_study(
    train_pos: List[str],
    train_neg: List[str],
    test_pos: List[str],
    test_neg: List[str],
    window: int = DONOR_WINDOW,
) -> None:
    configs = [
        ("One-hot only", ["one_hot"]),
        ("k-mer (2+3)", ["kmer2", "kmer3"]),
        ("Dinuc positional", ["dinuc_pos"]),
        ("PWM log-odds", ["pwm"]),
        ("Chi2 dependency pairs", ["chi2_pairs"]),
        ("EBN LLR only", ["ebn_llr"]),
        ("One-hot + k-mer", ["one_hot", "kmer2", "kmer3"]),
        ("One-hot + dinuc + PWM", ["one_hot", "dinuc_pos", "pwm"]),
        ("Dependency-aware set", ["one_hot", "dinuc_pos", "chi2_pairs", "ebn_llr"]),
        ("Combined (all)", list(FeatureExtractor.ALL_FEATURES)),
    ]

    test_seqs = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [-1] * len(test_neg)

    print("\n-- Feature ablation (RBF-SVM, C=1.0) --------------------")
    print(f"{'Feature set':<30} {'Dim':>5} {'AUC-ROC':>9} {'MCC':>9} {'F1':>9}")
    print("-" * 66)

    for name, fts in configs:
        m = SVMSpliceSite(window=window, kernel="rbf", C=1.0, feature_set=fts)
        m.train(train_pos, train_neg)
        scores = m.decision_score_batch(test_seqs)
        preds = [1 if s >= 0 else -1 for s in scores]
        met = evaluate_full(test_labels, preds, scores)
        dim = m.extractor.feature_dim if m.extractor is not None else 0
        print(f"  {name:<28} {dim:>5} {met['AUC']:>9.4f} {met['MCC']:>9.4f} {met['F1']:>9.4f}")

    print("-" * 66)


def kernel_comparison(
    train_pos: List[str],
    train_neg: List[str],
    test_pos: List[str],
    test_neg: List[str],
    window: int = DONOR_WINDOW,
) -> None:
    kernels = [
        ("Linear", dict(kernel="linear", C=1.0)),
        ("RBF C=0.1", dict(kernel="rbf", C=0.1, gamma="scale")),
        ("RBF C=1.0", dict(kernel="rbf", C=1.0, gamma="scale")),
        ("RBF C=10", dict(kernel="rbf", C=10.0, gamma="scale")),
        ("Poly d=2", dict(kernel="poly", C=1.0, degree=2)),
        ("Poly d=3", dict(kernel="poly", C=1.0, degree=3)),
    ]

    test_seqs = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [-1] * len(test_neg)

    print("\n-- Kernel comparison (combined features) -----------------")
    print(f"{'Kernel / C':^18} {'Sens':>8} {'Spec':>8} {'AUC':>8} {'MCC':>8} {'F1':>8}")
    print("-" * 66)

    for name, kwargs in kernels:
        kernel = str(kwargs.get("kernel", "rbf"))
        C = float(kwargs.get("C", 1.0))
        gamma = kwargs.get("gamma", "scale")
        degree = int(kwargs.get("degree", 3))
        m = SVMSpliceSite(
            window=window,
            kernel=kernel,  # type: ignore[arg-type]
            C=C,
            gamma=gamma,  # type: ignore[arg-type]
            degree=degree,
            feature_set=list(FeatureExtractor.ALL_FEATURES),
        )
        m.train(train_pos, train_neg)
        scores = m.decision_score_batch(test_seqs)
        preds = [1 if s >= 0 else -1 for s in scores]
        met = evaluate_full(test_labels, preds, scores)
        print(
            f"  {name:<16} {met['sensitivity']:>8.4f} {met['specificity']:>8.4f}"
            f" {met['AUC']:>8.4f} {met['MCC']:>8.4f} {met['F1']:>8.4f}"
        )

    print("-" * 66)


def _build_probabilistic_models(train_pos, train_neg, window):
    class _PWM:
        def __init__(self, w):
            self.w = w
            self.lo: Optional[List[Dict[str, float]]] = None

        def train(self, pos, neg):
            n = self.w
            bg = {b: 0.25 for b in BASES}
            ct = [{b: PSEUDOCOUNT for b in BASES} for _ in range(n)]
            for s in validate_seqs(pos, n):
                for i, c in enumerate(s):
                    ct[i][c] += 1
            self.lo = []
            for d in ct:
                tot = sum(d.values())
                self.lo.append({b: math.log(max(d[b] / tot, 1e-10) / bg[b]) for b in BASES})

        def score(self, seq):
            if self.lo is None:
                raise RuntimeError("Model not trained.")
            return sum(self.lo[i].get(c, 0) for i, c in enumerate(seq.upper()) if c in BASE_IDX)

        def score_batch(self, seqs):
            return [self.score(s) for s in seqs]

    class _WAM:
        def __init__(self, w):
            self.w = w
            self.first: Optional[Dict[str, float]] = None
            self.cpt: Optional[List[Dict[str, Dict[str, float]]]] = None

        def train(self, pos, neg):
            valid = validate_seqs(pos, self.w)
            fc = {b: PSEUDOCOUNT for b in BASES}
            for s in valid:
                fc[s[0]] += 1
            tot = sum(fc.values())
            self.first = {b: fc[b] / tot for b in BASES}
            cond = [{p: {c: PSEUDOCOUNT for c in BASES} for p in BASES} for _ in range(self.w)]
            for s in valid:
                for i in range(1, self.w):
                    cond[i][s[i - 1]][s[i]] += 1
            self.cpt = []
            for i, d in enumerate(cond):
                row = {}
                for p in BASES:
                    t = sum(d[p].values())
                    row[p] = {c: d[p][c] / t for c in BASES}
                self.cpt.append(row)

        def score(self, seq):
            if self.first is None or self.cpt is None:
                raise RuntimeError("Model not trained.")
            seq = seq.upper()
            s = math.log(max(self.first.get(seq[0], 1e-10), 1e-10) / 0.25)
            for i in range(1, self.w):
                p = self.cpt[i][seq[i - 1]][seq[i]]
                s += math.log(max(p, 1e-10) / 0.25)
            return s

        def score_batch(self, seqs):
            return [self.score(s) for s in seqs]

    wmm = _PWM(window)
    wmm.train(train_pos, train_neg)
    wam = _WAM(window)
    wam.train(train_pos, train_neg)
    return wmm, wam


def _load_task2_bn_model_class() -> Optional[type]:
    task2_dir = Path(__file__).resolve().parent.parent / "Task2 Bayesian network splice"
    model_path = task2_dir / "splice_model.py"
    utils_path = task2_dir / "splice_utils.py"
    if not model_path.exists():
        return None
    if not utils_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("task2_splice_model", str(model_path))
    if spec is None or spec.loader is None:
        return None

    utils_spec = importlib.util.spec_from_file_location("splice_utils", str(utils_path))
    if utils_spec is None or utils_spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    utils_mod = importlib.util.module_from_spec(utils_spec)
    original_sys_path = list(sys.path)
    original_splice_utils = sys.modules.get("splice_utils")
    task3_dir = str(Path(__file__).resolve().parent)
    try:
        sys.path = [str(task2_dir)] + [p for p in sys.path if p != task3_dir]
        sys.modules["splice_utils"] = utils_mod
        utils_spec.loader.exec_module(utils_mod)
        spec.loader.exec_module(mod)
    finally:
        sys.path = original_sys_path
        if original_splice_utils is not None:
            sys.modules["splice_utils"] = original_splice_utils
        else:
            sys.modules.pop("splice_utils", None)
    return getattr(mod, "BayesianNetworkModel", None)


def four_way_comparison(
    train_pos: List[str],
    train_neg: List[str],
    test_pos: List[str],
    test_neg: List[str],
    window: int = DONOR_WINDOW,
    plot_output: Optional[str | Path] = None,
) -> None:
    test_seqs = test_pos + test_neg
    test_labels_pm1 = [1] * len(test_pos) + [-1] * len(test_neg)

    svm = SVMSpliceSite(window=window, kernel="rbf", C=1.0, feature_set=list(FeatureExtractor.ALL_FEATURES))
    svm.train(train_pos, train_neg)

    svm_sc = svm.decision_score_batch(test_seqs)

    models = [
        ("SVM (RBF)", svm_sc, test_labels_pm1),
    ]

    print("\n" + "=" * 68)
    print("  Support Vector Machine - Donor Site Prediction")
    print("=" * 68)
    print(f"  {'Metric':<16}", end="")
    for name, _, _ in models:
        print(f"{name:>14}", end="")
    print()
    print("  " + "-" * 64)

    rows = [
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Precision", "precision"),
        ("F1 score", "F1"),
        ("MCC", "MCC"),
        ("AUC-ROC", "AUC"),
    ]

    all_mets = []
    roc_data = []
    for name, sc, lbl in models:
        preds = [1 if s >= 0.0 else -1 for s in sc]
        met = evaluate_full(lbl, preds, sc)
        all_mets.append(met)
        fpr = []
        tpr = []
        thresholds = sorted(set(sc), reverse=True)
        if thresholds:
            step = max(1, len(thresholds) // 200)
            thresholds = thresholds[::step]
        fpr.append(0.0)
        tpr.append(0.0)
        for thr in thresholds:
            pred = [1 if s >= thr else -1 for s in sc]
            m = evaluate_full(lbl, pred, sc)
            fpr.append(1.0 - m["specificity"])
            tpr.append(m["sensitivity"])
        fpr.append(1.0)
        tpr.append(1.0)
        auc = roc_auc_score(lbl, sc)
        roc_data.append((name, fpr, tpr, float(auc)))

    for label, key in rows:
        vals = [m[key] for m in all_mets]
        best = max(vals)
        print(f"  {label:<16}", end="")
        for v in vals:
            mark = "<" if abs(v - best) < 1e-6 else " "
            print(f"{v:>13.4f}{mark}", end="")
        print()

    print("  " + "-" * 64)
    print("  < = best in row")
    print("=" * 68)

    if plot_output is not None:
        saved = plot_roc_curves(roc_data, plot_output)
        print(f"\n  ROC figure saved to: {saved}")
