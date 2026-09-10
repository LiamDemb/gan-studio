"""Distribution distance from USER-SUPPLIED features. No pretrained weights.

This is FID only when both inputs use the same specified Inception feature
extractor, preprocessing and sample protocol. Arbitrary features are not FID.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def feature_stats(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite [N,D] matrix with N >= 2")
    return x.mean(0), np.atleast_2d(np.cov(x, rowvar=False))


def frechet_distance(real, fake):
    mr, cr = feature_stats(real)
    mf, cf = feature_stats(fake)
    if mr.shape != mf.shape:
        raise ValueError("Feature dimensions differ")
    # Symmetric PSD sandwich avoids complex square roots of Cr @ Cf.
    eigenvalues, eigenvectors = np.linalg.eigh(cr)
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    sandwich = root @ cf @ root
    ev = np.linalg.eigvalsh((sandwich + sandwich.T) / 2)
    distance = (
        (mr - mf) @ (mr - mf)
        + np.trace(cr)
        + np.trace(cf)
        - 2 * np.sqrt(np.maximum(ev, 0)).sum()
    )
    return max(float(distance), 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_features")
    parser.add_argument("fake_features")
    parser.add_argument("--extractor-description", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    real, fake = [
        np.load(p, allow_pickle=False) for p in (args.real_features, args.fake_features)
    ]
    result = {
        "metric": "Frechet distance of supplied features",
        "value": frechet_distance(real, fake),
        "real_samples": len(real),
        "fake_samples": len(fake),
        "dimensions": real.shape[1],
        "extractor": args.extractor_description,
        "fid_protocol_verified": False,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
