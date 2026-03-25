from wam_models import WMMModel, WAMModel, evaluate, roc_auc, DONOR_WINDOW
from data_io import (
    load_txt_sequences_from_dir,
    make_donor_positive,
    make_donor_negative,
)
from scanner import SpliceSiteScanner
import pickle


def compare_models(
    train_pos,
    train_neg,
    test_pos,
    test_neg,
    window=DONOR_WINDOW,
    threshold=0.0,
):
    wmm = WMMModel(window=window, site="donor")
    wmm.train(train_pos, train_neg)

    wam = WAMModel(window=window, site="donor")
    wam.train(train_pos, train_neg)

    test_seqs = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)

    wmm_scores = wmm.score_batch(test_seqs)
    wam_scores = wam.score_batch(test_seqs)

    wmm_preds = [int(s >= threshold) for s in wmm_scores]
    wam_preds = [int(s >= threshold) for s in wam_scores]

    wmm_m = evaluate(test_labels, wmm_preds)
    wam_m = evaluate(test_labels, wam_preds)
    _, _, wmm_auc = roc_auc(test_labels, wmm_scores)
    _, _, wam_auc = roc_auc(test_labels, wam_scores)

    print("\n" + "=" * 60)
    print("  WMM vs WAM — Donor Site Prediction (5′ splice sites)")
    print("=" * 60)
    print(f"{'Metric':<18} {'WMM':>10} {'WAM':>10}")
    print("-" * 40)

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
            w_val, a_val = wmm_auc, wam_auc
        else:
            w_val, a_val = wmm_m[key], wam_m[key]
        marker = " ◀" if a_val > w_val else ""
        print(f"  {label:<16} {w_val:>10.4f} {a_val:>10.4f}{marker}")

    print("-" * 40)
    print("  ◀ = WAM outperforms WMM")
    print("=" * 60)

    wam.print_summary()


def train_and_save_model(train_pos, train_neg, model_path="wam_model.pkl"):
    """Train WAM model on training data and save to file."""
    wam = WAMModel(window=DONOR_WINDOW, site="donor")
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


def demo() -> None:
    training_dir = "/Users/shiroko/FlutterProjects/BDM Homework/Training and testing datasets/Training Set"
    testing_dir = "/Users/shiroko/FlutterProjects/BDM Homework/Training and testing datasets/Testing Set"

    train_pos, train_neg = load_txt_sequences_from_dir(training_dir, window=DONOR_WINDOW, site="donor")
    test_pos, test_neg = load_txt_sequences_from_dir(testing_dir, window=DONOR_WINDOW, site="donor")

    if not train_pos or not test_pos:
        print("Training/testing directories not found or empty; generating synthetic dataset …")
        all_pos = make_donor_positive(1000)
        all_neg = make_donor_negative(1000)

        split = 800
        train_pos, test_pos = all_pos[:split], all_pos[split:]
        train_neg, test_neg = all_neg[:split], all_neg[split:]

    if not train_neg:
        print("Warning: train_neg is empty; using synthetic negatives")
        train_neg = make_donor_negative(len(train_pos))

    if not test_neg:
        print("Warning: test_neg is empty; splitting test_pos into negatives")
        all_test = test_pos
        half = len(all_test) // 2
        test_neg = all_test[:half]
        test_pos = all_test[half:]

    compare_models(train_pos, train_neg, test_pos, test_neg)

    # Train and save model
    model_path = "wam_model.pkl"
    train_and_save_model(train_pos, train_neg, model_path)

    print("\n── Genome scan demo ──────────────────────────────")
    # Load model and predict on sequence
    genome = ("ATCGATCGATCG"
              "CAGGTAAGTATCG"
              "GCATCGATCGATCG"
              "AAGGTAAGTGCTA"
              "TTTGCATCGATCG")

    hits = predict_on_sequence(genome, model_path, threshold=1.0)
    print(f"Genome length: {len(genome)} bp")
    for pos, sc in hits:
        context = genome[max(0, pos-2): pos+5]
        print(f"  position {pos:>4}  score={sc:+.3f}  context: …{context}…")


if __name__ == "__main__":
    demo()
