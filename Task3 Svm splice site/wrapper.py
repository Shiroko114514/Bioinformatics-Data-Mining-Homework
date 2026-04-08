import pickle
from pathlib import Path
from typing import Union

from splice_data import load_real_dataset_split
from splice_features import FeatureExtractor
from splice_model import SVMSpliceSite
from splice_utils import DONOR_WINDOW

DEFAULT_MODEL_PATH = Path(__file__).with_name("svm_splice_site.pkl")


def train_and_save_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Path:
    train_pos, train_neg, test_pos, test_neg = load_real_dataset_split(site="donor")

    print("Loaded dataset:")
    print(f"  Train positives: {len(train_pos)}")
    print(f"  Train negatives: {len(train_neg)}")
    print(f"  Test positives : {len(test_pos)}")
    print(f"  Test negatives : {len(test_neg)}")

    model = SVMSpliceSite(
        window=DONOR_WINDOW,
        kernel="rbf",
        feature_set=list(FeatureExtractor.ALL_FEATURES),
    )
    model.train(train_pos, train_neg)

    model_path = Path(model_path)
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")
    return model_path


def main() -> None:
    train_and_save_model()


if __name__ == "__main__":
    main()
