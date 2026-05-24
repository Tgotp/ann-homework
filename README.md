# ANN Parallel Search Homework

This repository contains the single-node parallel implementation for the ANN homework.

Implemented search modes are configured in `files/ann_config.txt`:

- `mode=0`: Flat baseline
- `mode=1`: Flat SIMD
- `mode=2`: Flat OpenMP
- `mode=3`: Flat OpenMP + SIMD
- `mode=4`: SQ-SIMD with top-p reranking
- `mode=5`: PQ-SIMD with top-p reranking
- `mode=6`: PQ-SIMD + OpenMP with top-p reranking

The implementation targets the course ARM server and uses NEON on AArch64, with x86 SIMD/scalar fallbacks guarded by compile-time macros.

## Files

- `main.cc`: benchmark entry and dispatch integration
- `ann_search_versions.h`: Flat/SQ/PQ SIMD and OpenMP variants
- `flat_scan.h`: original flat baseline
- `files/ann_config.txt`: runtime experiment configuration
- `run_experiments.sh`: batch experiment runner
- `experiment_results/summary.tsv`: collected experiment results

## Run

On the course server, edit `files/ann_config.txt`, then run:

```sh
bash test.sh 1 1
```

To run the configured experiment suite:

```sh
bash run_experiments.sh
```
