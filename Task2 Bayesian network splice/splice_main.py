import argparse
from pathlib import Path

from splice_data import load_positive_sites_from_dir, load_sequences_from_dir, generate_negative_samples
from splice_model import BayesianNetworkModel, BNScanner
from splice_eval import compare_models
from splice_utils import DONOR_WINDOW, ACCEPTOR_WINDOW


def train_model_for_prediction(use_real_data: bool = True,
                               site_type: str = 'donor',
                               window: int = None,
                               structure: str = 'chow-liu',
                               max_parents: int = 2,
                               dependency_threshold: float = 6.0) -> BayesianNetworkModel:
    if window is None:
        window = DONOR_WINDOW if site_type == 'donor' else ACCEPTOR_WINDOW

    if use_real_data:
        base_path = Path(__file__).parent.parent / 'Training and testing datasets'
        train_dir = base_path / 'Training Set'
        train_pos = load_positive_sites_from_dir(str(train_dir), site_type=site_type, window=window)
        train_seqs = load_sequences_from_dir(str(train_dir))
        train_neg = generate_negative_samples(train_seqs, len(train_pos), window=window, exclude_sites=set(train_pos))
    else:
        raise NotImplementedError('Synthetic training not implemented in main module')

    model = BayesianNetworkModel(
        window=window,
        site=site_type,
        structure=structure,
        max_parents=max_parents,
        dependency_threshold=dependency_threshold,
    )
    model.train(train_pos, train_neg)
    return model


def predict_splice_sites_in_genome(seq: str, model: BayesianNetworkModel, threshold: float = 0.0):
    scanner = BNScanner(model, threshold=threshold)
    hits = scanner.scan(seq)
    out = []
    for pos, score in hits:
        off = scanner.offset
        w = model.window
        start = max(0, pos - off)
        window_seq = seq[start:start + w]
        out.append((pos, score, window_seq))
    return out


def main():
    parser = argparse.ArgumentParser(description='Splice site predictor')
    parser.add_argument('--predict', type=str, help='Input genome sequence text to predict')
    parser.add_argument('--site', choices=['donor', 'acceptor'], default='donor')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--window', type=int, choices=[9, 23], default=None)
    parser.add_argument('--structure', choices=['chain', 'chow-liu', 'ebn'], default='chow-liu')
    parser.add_argument('--max-parents', type=int, default=2)
    parser.add_argument('--chi2-threshold', type=float, default=6.0)
    parser.add_argument('--no-real-data', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    if args.predict:
        model = train_model_for_prediction(use_real_data=not args.no_real_data,
                                           site_type=args.site,
                                           window=args.window,
                                           structure=args.structure,
                                           max_parents=args.max_parents,
                                           dependency_threshold=args.chi2_threshold)
        results = predict_splice_sites_in_genome(args.predict.upper(), model, threshold=args.threshold)
        if not results:
            print('No sites above threshold.')
        else:
            print('pos\tscore\twindow')
            for pos, score, window_seq in results:
                print(f'{pos}\t{score:.4f}\t{window_seq}')
        return

    if args.demo or not args.predict:
        print('Running benchmark demo with training/testing data output...')
        base_path = Path(__file__).parent.parent / 'Training and testing datasets'
        train_dir = str(base_path / 'Training Set')
        test_dir = str(base_path / 'Testing Set')

        train_pos = load_positive_sites_from_dir(train_dir, site_type=args.site, window=(args.window or (DONOR_WINDOW if args.site == 'donor' else ACCEPTOR_WINDOW)))
        test_pos = load_positive_sites_from_dir(test_dir, site_type=args.site, window=(args.window or (DONOR_WINDOW if args.site == 'donor' else ACCEPTOR_WINDOW)))
        train_seqs = load_sequences_from_dir(train_dir)
        test_seqs = load_sequences_from_dir(test_dir)

        train_neg = generate_negative_samples(train_seqs, len(train_pos), window=(args.window or (DONOR_WINDOW if args.site == 'donor' else ACCEPTOR_WINDOW)), exclude_sites=set(train_pos))
        test_neg = generate_negative_samples(test_seqs, len(test_pos), window=(args.window or (DONOR_WINDOW if args.site == 'donor' else ACCEPTOR_WINDOW)), exclude_sites=set(test_pos))

        compare_models(train_pos, train_neg, test_pos, test_neg,
                       window=args.window or (DONOR_WINDOW if args.site == 'donor' else ACCEPTOR_WINDOW),
                       threshold=args.threshold,
                       plot_output=Path(__file__).with_name('roc_curves.svg'))


if __name__ == '__main__':
    main()
