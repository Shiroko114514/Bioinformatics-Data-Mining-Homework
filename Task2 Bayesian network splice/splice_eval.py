import math
import random
from html import escape
from pathlib import Path
from typing import List, Tuple, Dict
from splice_model import _WMMModel, _WAMModel, BayesianNetworkModel
from roc_plot import plot_roc_curves


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


def plot_roc_curves(
    roc_data: List[Tuple[str, List[float], List[float], float]],
    output_path: str | Path,
    title: str = 'ROC Curves for Splice Site Prediction',
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    width = 920
    height = 680
    margin_left = 90
    margin_right = 260
    margin_top = 70
    margin_bottom = 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    def sx(x: float) -> float:
        return margin_left + x * plot_w

    def sy(y: float) -> float:
        return margin_top + (1.0 - y) * plot_h

    elements = []
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    elements.append(
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700">{escape(title)}</text>'
    )

    # Axes and grid.
    elements.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="1.5"/>')
    elements.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="1.5"/>')
    for tick in range(6):
        x = tick / 5.0
        y = tick / 5.0
        px = sx(x)
        py = sy(y)
        elements.append(f'<line x1="{px:.1f}" y1="{margin_top}" x2="{px:.1f}" y2="{margin_top + plot_h}" stroke="#e6e6e6" stroke-width="1"/>')
        elements.append(f'<line x1="{margin_left}" y1="{py:.1f}" x2="{margin_left + plot_w}" y2="{py:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        elements.append(f'<text x="{px:.1f}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">{x:.1f}</text>')
        elements.append(f'<text x="{margin_left - 12}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12">{y:.1f}</text>')

    # Random baseline.
    elements.append(
        f'<line x1="{sx(0.0):.1f}" y1="{sy(0.0):.1f}" x2="{sx(1.0):.1f}" y2="{sy(1.0):.1f}" '
        f'stroke="#777" stroke-width="1.5" stroke-dasharray="6,5"/>'
    )

    legend_x = width - margin_right + 20
    legend_y = margin_top + 10
    elements.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="700">Legend</text>')
    legend_y += 18

    for idx, (name, fpr, tpr, auc) in enumerate(roc_data):
        color = palette[idx % len(palette)]
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(fpr, tpr))
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'
        )
        elements.append(f'<circle cx="{sx(fpr[-1]):.1f}" cy="{sy(tpr[-1]):.1f}" r="3.5" fill="{color}"/>')
        elements.append(
            f'<line x1="{legend_x}" y1="{legend_y - 7}" x2="{legend_x + 28}" y2="{legend_y - 7}" stroke="{color}" stroke-width="3"/>'
        )
        label = f'{name} (AUC={auc:.4f})'
        elements.append(
            f'<text x="{legend_x + 36}" y="{legend_y - 2}" font-family="Arial, Helvetica, sans-serif" font-size="12">{escape(label)}</text>'
        )
        legend_y += 22

    elements.append(
        f'<text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13">False Positive Rate</text>'
    )
    elements.append(
        f'<text x="22" y="{height / 2:.1f}" transform="rotate(-90 22 {height / 2:.1f})" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13">True Positive Rate</text>'
    )

    svg = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    out.write_text(svg, encoding='utf-8')
    return out


def compare_models(train_pos: List[str], train_neg: List[str], test_pos: List[str], test_neg: List[str],
                   window: int = 9, threshold: float = 0.0,
                   plot_output: str | Path | None = None) -> None:
    models = [
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
    print("  Bayesian Network Methods — Donor Site Prediction")
    print("="*70)
    hdr = f"{'Metric':<20}" + ''.join(f"{n:>15}" for n, _ in models)
    print(hdr)
    print("─"*80)
    result_rows = []
    aucs = []
    roc_data = []
    for name, m in models:
        scores = m.score_batch(test_seqs)
        preds = [int(s >= threshold) for s in scores]
        met = evaluate(test_labels, preds)
        fpr, tpr, auc = roc_auc(test_labels, scores)
        result_rows.append((met, auc))
        aucs.append(auc)
        roc_data.append((name, fpr, tpr, auc))
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

    if plot_output is not None:
        saved = plot_roc_curves(roc_data, plot_output)
        print(f"\n  ROC figure saved to: {saved}")

    bn_cl = models[0][1]
    bn_cl.print_summary()
    non_adj = bn_cl.non_adjacent_edges()
    if non_adj:
        print("\n  Long-range edges discovered by Chow-Liu (not captured by WAM):")
        for child, parent in non_adj:
            mi_val = bn_cl.mi_matrix[child][parent] if bn_cl.mi_matrix else float('nan')
            print(f"    pos {parent} → pos {child}  gap={abs(child-parent)}  MI={mi_val:.4f} nats")

    bn_ebn = models[1][1]
    bn_ebn.print_summary()
    non_adj_ebn = bn_ebn.non_adjacent_edges()
    if non_adj_ebn:
        print("\n  Long-range edges discovered by EBN dependency expansion:")
        for child, parent in non_adj_ebn[:20]:
            print(f"    pos {parent} → pos {child}  gap={abs(child-parent)}")
