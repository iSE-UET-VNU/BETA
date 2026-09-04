#!/usr/bin/env bash
# Rebuild the performance matrix and similarity encoder from scratch on your own dataset list.
set -euo pipefail

beta build-matrix --datasets ids.txt --out assets/performance_matrix.csv
beta train-similarity --matrix assets/performance_matrix.csv --out assets/similarity_encoder.pt
