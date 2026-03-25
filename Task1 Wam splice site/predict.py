#!/usr/bin/env python3
"""
Predict splice sites in an input DNA sequence using a trained WAM model.
Usage: python predict.py "ATCGATCG..." [model_path] [threshold]
"""

import sys
from main import predict_on_sequence

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <sequence> [model_path=wam_model.pkl] [threshold=1.0]")
        sys.exit(1)

    sequence = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "wam_model.pkl"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    hits = predict_on_sequence(sequence, model_path, threshold)

    print(f"Sequence length: {len(sequence)} bp")
    if hits:
        print("Predicted splice sites:")
        for pos, sc in hits:
            context = sequence[max(0, pos-2): pos+5]
            print(f"  position {pos:>4}  score={sc:+.3f}  context: …{context}…")
    else:
        print("No splice sites predicted above threshold.")

if __name__ == "__main__":
    main()