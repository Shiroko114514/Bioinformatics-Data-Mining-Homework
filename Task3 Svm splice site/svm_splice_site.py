"""Compatibility facade for split SVM splice-site modules."""

from splice_data import (
	extract_splice_sites_from_ranges,
	generate_negative_samples,
	load_positive_sites_from_dir,
	load_real_dataset_split,
	load_sequences_from_dir,
	make_donor_negative,
	make_donor_positive,
	parse_genbank_file,
)
from splice_eval import ablation_study, evaluate_full, four_way_comparison, kernel_comparison
from splice_features import FeatureExtractor
from splice_main import demo
from splice_model import SVMSpliceSite
from splice_utils import (
	ACCEPTOR_AG_POS,
	ACCEPTOR_WINDOW,
	BASE_IDX,
	BASES,
	DONOR_GT_POS,
	DONOR_WINDOW,
	PSEUDOCOUNT,
	validate_seqs,
)

__all__ = [
	"BASES",
	"BASE_IDX",
	"PSEUDOCOUNT",
	"DONOR_WINDOW",
	"ACCEPTOR_WINDOW",
	"DONOR_GT_POS",
	"ACCEPTOR_AG_POS",
	"validate_seqs",
	"parse_genbank_file",
	"extract_splice_sites_from_ranges",
	"load_positive_sites_from_dir",
	"load_sequences_from_dir",
	"generate_negative_samples",
	"load_real_dataset_split",
	"make_donor_positive",
	"make_donor_negative",
	"FeatureExtractor",
	"SVMSpliceSite",
	"evaluate_full",
	"ablation_study",
	"kernel_comparison",
	"four_way_comparison",
	"demo",
]


if __name__ == "__main__":
	demo()
