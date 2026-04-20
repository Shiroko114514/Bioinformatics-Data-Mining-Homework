import random
from pathlib import Path

import numpy as np

from splice_data import load_real_dataset_split
from splice_eval import ablation_study, four_way_comparison, kernel_comparison
from splice_features import FeatureExtractor
from splice_model import SVMSpliceSite
from splice_utils import DONOR_WINDOW


def demo() -> None:
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("  SVM Splice Site Predictor - Demo")
    print("=" * 60)
    print("Loading provided training/testing datasets ...\n")

    train_pos, train_neg, test_pos, test_neg = load_real_dataset_split(site="donor")

    print(f"  Train positives: {len(train_pos)}")
    print(f"  Train negatives: {len(train_neg)}")
    print(f"  Test positives : {len(test_pos)}")
    print(f"  Test negatives : {len(test_neg)}\n")

    ablation_study(train_pos, train_neg, test_pos, test_neg)
    kernel_comparison(train_pos, train_neg, test_pos, test_neg)
    four_way_comparison(
        train_pos,
        train_neg,
        test_pos,
        test_neg,
        plot_output=Path(__file__).with_name("roc_curves.svg"),
    )

    print("\n-- Linear SVM - top 12 discriminative features ------------")
    lin = SVMSpliceSite(window=DONOR_WINDOW, kernel="linear", C=1.0, feature_set=list(FeatureExtractor.ALL_FEATURES))
    lin.train(train_pos, train_neg)
    lin.print_summary()
    top = lin.top_features(12)
    print(f"\n  {'Feature':<26} {'Weight':>10}")
    print("  " + "-" * 38)
    for name, w in top:
        bar = "+" * min(int(abs(w) * 6), 20) if w > 0 else "-" * min(int(abs(w) * 6), 20)
        sign = "+" if w > 0 else " "
        print(f"  {name:<26} {sign}{abs(w):>8.4f}  {bar}")

    print("\n-- 5-fold cross-validation (RBF, combined features) ------")
    rbf = SVMSpliceSite(window=DONOR_WINDOW, kernel="rbf", C=1.0, feature_set=list(FeatureExtractor.ALL_FEATURES))
    cv_res = rbf.cross_validate(train_pos, train_neg, n_folds=5)
    print(f"  AUC-ROC : {cv_res['auc_mean']:.4f} +/- {cv_res['auc_std']:.4f}")
    print(f"  MCC     : {cv_res['mcc_mean']:.4f} +/- {cv_res['mcc_std']:.4f}")
    print(f"  F1      : {cv_res['f1_mean']:.4f}  +/- {cv_res['f1_std']:.4f}")

    print("\n-- Genome scan demo (RBF-SVM, threshold=0.5) --------------")
    rbf.train(train_pos, train_neg)
    genome = (
        "ATCGATCGATCG"
        "CAGGTAAGTATCG"
        "GCATCGATCGATCG"
        "AAGGTAAGTGCTA"
        "TTTGCATCGATCG"
    )
    hits = rbf.scan(genome, threshold=0.5)
    print(f"  Genome: {len(genome)} bp | threshold=0.5")
    for pos, sc in hits:
        ctx = genome[max(0, pos - 2): pos + 5]
        print(f"  pos={pos:>4}  score={sc:+.3f}  context: ...{ctx}...")
    if not hits:
        print("  (none above threshold)")


if __name__ == "__main__":
    demo()
