import pickle
import sys
from pathlib import Path
from typing import Union

from splice_model import SVMSpliceSite

DEFAULT_MODEL_PATH = Path(__file__).with_name("svm_splice_site.pkl")
DEFAULT_THRESHOLD = 0.0


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> SVMSpliceSite:
    with Path(model_path).open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, SVMSpliceSite):
        raise TypeError("Loaded object is not an SVMSpliceSite model.")
    return model


def predict_sequence(sequence: str, model_path: str | Path = DEFAULT_MODEL_PATH, threshold: float = DEFAULT_THRESHOLD) -> None:
    model = load_model(model_path)
    genome = sequence.upper().strip()
    hits = model.scan(genome, threshold=threshold)

    if not hits:
        print("No sites above threshold.")
        return

    print("pos\tscore\twindow")
    for pos, score in hits:
        off = 3 if model.site == "donor" else 20
        start = max(0, pos - off)
        end = start + model.window
        window_seq = genome[start:end]
        print(f"{pos}\t{score:.4f}\t{window_seq}")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python predict.py "<DNA sequence>" [model_path] [threshold]')
        raise SystemExit(1)

    sequence = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL_PATH
    threshold = float(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_THRESHOLD
    predict_sequence(sequence, model_path=model_path, threshold=threshold)


if __name__ == "__main__":
    main()
