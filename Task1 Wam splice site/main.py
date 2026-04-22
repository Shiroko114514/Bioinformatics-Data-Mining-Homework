import argparse
import pickle
from pathlib import Path

from data_io import (
    load_strict_dataset_split,
    make_acceptor_negative,
    make_acceptor_positive,
    make_donor_negative,
    make_donor_positive,
)
from scanner import SpliceSiteScanner
from wam_models import (
    ACCEPTOR_WINDOW,
    DONOR_WINDOW,
    DependencyWAMModel,
    WAMModel,
    WMMModel,
    evaluate,
    roc_auc,
)
from roc_plot import plot_roc_curves


def _default_window(site: str, window: int | None) -> int:
    return window if window is not None else (DONOR_WINDOW if site == "donor" else ACCEPTOR_WINDOW)


def compare_models(
    train_pos,
    train_neg,
    test_pos,
    test_neg,
    window=DONOR_WINDOW,
    site="donor",
    dependency_threshold=6.0,
    max_dependency_pairs=16,
    threshold=0.0,
    plot_output=None,
):
    wmm = WMMModel(window=window, site=site)
    wmm.train(train_pos, train_neg)

    wam = WAMModel(window=window, site=site)
    wam.train(train_pos, train_neg)

    dep_wam = DependencyWAMModel(
        window=window,
        site=site,
        dependency_threshold=dependency_threshold,
        max_dependency_pairs=max_dependency_pairs,
    )
    dep_wam.train(train_pos, train_neg)

    test_seqs = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)

    wmm_scores = wmm.score_batch(test_seqs)
    wam_scores = wam.score_batch(test_seqs)
    dep_scores = dep_wam.score_batch(test_seqs)

    wmm_preds = [int(s >= threshold) for s in wmm_scores]
    wam_preds = [int(s >= threshold) for s in wam_scores]
    dep_preds = [int(s >= threshold) for s in dep_scores]

    wmm_m = evaluate(test_labels, wmm_preds)
    wam_m = evaluate(test_labels, wam_preds)
    dep_m = evaluate(test_labels, dep_preds)
    wmm_fpr, wmm_tpr, wmm_auc = roc_auc(test_labels, wmm_scores)
    wam_fpr, wam_tpr, wam_auc = roc_auc(test_labels, wam_scores)
    dep_fpr, dep_tpr, dep_auc = roc_auc(test_labels, dep_scores)

    print("\n" + "=" * 72)
    print(f"  WMM vs WAM vs Dependency-WAM — {site.title()} Site Prediction")
    print("=" * 72)
    print(f"{'Metric':<18} {'WMM':>10} {'WAM':>10} {'Dep-WAM':>10}")
    print("-" * 54)

    metrics_to_show = [
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Precision", "precision"),
        ("F1 score", "F1"),
        ("MCC", "MCC"),
        ("AUC-ROC", "__auc__"),
    ]

    for label, key in metrics_to_show:
        if key == "__auc__":
            w_val, a_val, d_val = wmm_auc, wam_auc, dep_auc
        else:
            w_val, a_val, d_val = wmm_m[key], wam_m[key], dep_m[key]
        marker = " ◀" if d_val >= max(w_val, a_val) else ""
        print(f"  {label:<16} {w_val:>10.4f} {a_val:>10.4f} {d_val:>10.4f}{marker}")

    print("-" * 54)
    print("  ◀ = best in the row")
    print("=" * 72)

    if plot_output is not None:
        saved = plot_roc_curves(
            [
                ("WMM", wmm_fpr, wmm_tpr, wmm_auc),
                ("WAM", wam_fpr, wam_tpr, wam_auc),
                ("Dependency-WAM", dep_fpr, dep_tpr, dep_auc),
            ],
            plot_output,
        )
        print(f"ROC figure saved to: {saved}")

    wam.print_summary()
    dep_wam.print_summary()


def train_and_save_model(train_pos, train_neg, model_path="wam_model.pkl", window=DONOR_WINDOW, site="donor"):
    """Train WAM model on training data and save to file."""
    wam = WAMModel(window=window, site=site)
    wam.train(train_pos, train_neg)
    with open(model_path, "wb") as f:
        pickle.dump(wam, f)
    print(f"Model saved to {model_path}")
    return wam


def predict_on_sequence(sequence, model_path="wam_model.pkl", threshold=1.0):
    """Load trained model and predict splice sites in input sequence."""
    with open(model_path, "rb") as f:
        wam = pickle.load(f)
    scanner = SpliceSiteScanner(wam, threshold=threshold)
    hits = scanner.scan(sequence)
    return hits


def _load_dataset(site: str, window: int):
    train_pos, train_neg, test_pos, test_neg = load_strict_dataset_split(site=site, window=window)

    if not train_pos or not test_pos:
        print("Training/testing directories not found or empty; generating synthetic dataset …")
        if site == "donor":
            all_pos = make_donor_positive(1000)
            all_neg = make_donor_negative(1000)
        else:
            all_pos = make_acceptor_positive(1000)
            all_neg = make_acceptor_negative(1000)
        split = 800
        train_pos, test_pos = all_pos[:split], all_pos[split:]
        train_neg, test_neg = all_neg[:split], all_neg[split:]

    if not train_neg:
        print("Warning: train_neg is empty; using synthetic negatives")
        train_neg = make_donor_negative(len(train_pos)) if site == "donor" else make_acceptor_negative(len(train_pos))

    if not test_neg:
        print("Warning: test_neg is empty; splitting test_pos into negatives")
        all_test = test_pos
        half = len(all_test) // 2
        test_neg = all_test[:half]
        test_pos = all_test[half:]

    return train_pos, train_neg, test_pos, test_neg


def run_demo(
    site: str = "donor",
    window: int | None = None,
    threshold: float = 0.0,
    dependency_threshold: float = 6.0,
    max_dependency_pairs: int = 16,
) -> None:
    window = _default_window(site, window)
    train_pos, train_neg, test_pos, test_neg = _load_dataset(site, window)

    compare_models(
        train_pos,
        train_neg,
        test_pos,
        test_neg,
        window=window,
        site=site,
        dependency_threshold=dependency_threshold,
        max_dependency_pairs=max_dependency_pairs,
        threshold=threshold,
        plot_output=Path(__file__).with_name("roc_curves.svg"),
    )

    model_path = "wam_model.pkl"
    train_and_save_model(train_pos, train_neg, model_path, window=window, site=site)

    print("\n── Genome scan demo ──────────────────────────────")
    genome = (
        "ATCGATCGATCG"
        "CAGGTAAGTATCG"
        "GCATCGATCGATCG"
        "AAGGTAAGTGCTA"
        "TTTGCATCGATCG"
    )

    hits = predict_on_sequence(genome, model_path, threshold=1.0)
    print(f"Genome length: {len(genome)} bp")
    for pos, sc in hits:
        context = genome[max(0, pos - 2) : pos + 5]
        print(f"  position {pos:>4}  score={sc:+.3f}  context: …{context}…")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task1 WAM splice site demo")
    parser.add_argument("--site", choices=["donor", "acceptor"], default="donor")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--dependency-threshold", type=float, default=6.0)
    parser.add_argument("--max-dependency-pairs", type=int, default=16)
    parser.add_argument("--predict", type=str, help="Input genome sequence text to predict")
    parser.add_argument("--demo", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.predict:
        window = _default_window(args.site, args.window)
        train_pos, train_neg, _, _ = _load_dataset(args.site, window)
        model = train_and_save_model(train_pos, train_neg, "wam_model.pkl", window=window, site=args.site)
        results = predict_on_sequence(args.predict.upper(), "wam_model.pkl", threshold=args.threshold)
        if not results:
            print("No sites above threshold.")
        else:
            print("pos\tscore\twindow")
            for pos, score in results:
                start = max(0, pos - (3 if args.site == "donor" else 20))
                window_seq = args.predict.upper()[start : start + window]
                print(f"{pos}\t{score:.4f}\t{window_seq}")
        return

    if args.demo or not args.predict:
        run_demo(
            site=args.site,
            window=args.window,
            threshold=args.threshold,
            dependency_threshold=args.dependency_threshold,
            max_dependency_pairs=args.max_dependency_pairs,
        )


def demo() -> None:
    main(["--demo"])


if __name__ == "__main__":
    main()
