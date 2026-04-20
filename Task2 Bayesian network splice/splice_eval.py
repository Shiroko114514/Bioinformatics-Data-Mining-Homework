import math
import random
from typing import List, Tuple, Dict
from splice_model import _WMMModel, _WAMModel, BayesianNetworkModel


def evaluate(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(t == p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == p == 0 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                sensitivity=sens, specificity=spec,
                precision=prec, F1=f1, MCC=mcc)


def roc_auc(y_true: List[int], scores: List[float], n_thr: int = 200) -> Tuple[List[float], List[float], float]:
    thresholds = sorted(set(scores), reverse=True)
    step = max(1, len(thresholds) // n_thr)
    thresholds = thresholds[::step]
    fpr, tpr = [0.0], [0.0]
    for thr in thresholds:
        pred = [int(s >= thr) for s in scores]
        m = evaluate(y_true, pred)
        fpr.append(1.0 - m['specificity'])
        tpr.append(m['sensitivity'])
    fpr.append(1.0); tpr.append(1.0)
    auc = sum((fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2 for i in range(1, len(fpr)))
    return fpr, tpr, auc


def compare_models(train_pos: List[str], train_neg: List[str], test_pos: List[str], test_neg: List[str],
                   window: int = 9, threshold: float = 0.0) -> None:
    models = [
        ("WMM", _WMMModel(window)),
        ("WAM", _WAMModel(window)),
        ("BN Chow-Liu", BayesianNetworkModel(window, site='donor', structure='chow-liu')),
        ("BN EBN(p=2)", BayesianNetworkModel(window, site='donor', structure='ebn', max_parents=2)),
    ]
    print("\nTraining models …")
    for name, m in models:
        print(f"  {name} …", end=' ', flush=True)
        m.train(train_pos, train_neg)
        print('done')
    test_seqs = test_pos + test_neg
    test_labels = [1]*len(test_pos) + [0]*len(test_neg)
    print("\n" + "="*70)
    print("  WMM / WAM / BN Chow-Liu / BN EBN(p=2)  — Donor Site Prediction")
    print("="*70)
    hdr = f"{'Metric':<20}" + ''.join(f"{n:>15}" for n, _ in models)
    print(hdr)
    print("─"*80)
    result_rows = []
    aucs = []
    for name, m in models:
        scores = m.score_batch(test_seqs)
        preds = [int(s >= threshold) for s in scores]
        met = evaluate(test_labels, preds)
        _, _, auc = roc_auc(test_labels, scores)
        result_rows.append((met, auc))
        aucs.append(auc)
    metrics_to_show = [
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Precision", "precision"),
        ("F1 score", "F1"),
        ("MCC", "MCC"),
        ("AUC-ROC", None),
    ]
    for label, key in metrics_to_show:
        vals = aucs if key is None else [row[0][key] for row in result_rows]
        best = max(vals)
        row = f"  {label:<18}"
        for v in vals:
            row += f"{v:>13.4f}" + ("◀" if abs(v - best) < 1e-6 else " ")
        print(row)
    print("─"*80)
    print("  ◀ = best in row")
    print("="*70)

    bn_cl = models[2][1]
    bn_cl.print_summary()
    non_adj = bn_cl.non_adjacent_edges()
    if non_adj:
        print("\n  Long-range edges discovered by Chow-Liu (not captured by WAM):")
        for child, parent in non_adj:
            mi_val = bn_cl.mi_matrix[child][parent] if bn_cl.mi_matrix else float('nan')
            print(f"    pos {parent} → pos {child}  gap={abs(child-parent)}  MI={mi_val:.4f} nats")

    bn_ebn = models[3][1]
    bn_ebn.print_summary()
    non_adj_ebn = bn_ebn.non_adjacent_edges()
    if non_adj_ebn:
        print("\n  Long-range edges discovered by EBN dependency expansion:")
        for child, parent in non_adj_ebn[:20]:
            print(f"    pos {parent} → pos {child}  gap={abs(child-parent)}")
