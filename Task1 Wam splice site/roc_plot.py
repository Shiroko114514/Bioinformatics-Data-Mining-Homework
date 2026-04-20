from html import escape
from pathlib import Path
from typing import List, Tuple, Union


def plot_roc_curves(
    roc_data: List[Tuple[str, List[float], List[float], float]],
    output_path: Union[str, Path],
    title: str = "ROC Curves for Splice Site Prediction",
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

    elements: List[str] = []
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    elements.append(
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700">{escape(title)}</text>'
    )

    elements.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" '
        f'y2="{margin_top + plot_h}" stroke="#222" stroke-width="1.5"/>'
    )
    elements.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{margin_top + plot_h}" stroke="#222" stroke-width="1.5"/>'
    )

    for tick in range(6):
        x = tick / 5.0
        y = tick / 5.0
        px = sx(x)
        py = sy(y)
        elements.append(f'<line x1="{px:.1f}" y1="{margin_top}" x2="{px:.1f}" y2="{margin_top + plot_h}" stroke="#e6e6e6" stroke-width="1"/>')
        elements.append(f'<line x1="{margin_left}" y1="{py:.1f}" x2="{margin_left + plot_w}" y2="{py:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        elements.append(f'<text x="{px:.1f}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">{x:.1f}</text>')
        elements.append(f'<text x="{margin_left - 12}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12">{y:.1f}</text>')

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
        label = f"{name} (AUC={auc:.4f})"
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
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    out.write_text(svg, encoding="utf-8")
    return out
