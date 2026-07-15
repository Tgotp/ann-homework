# GPU ANNS Flat Search

This directory contains a standalone CUDA implementation of exact flat
approximate-nearest-neighbor search for the course final report. It uses inner
product distance:

```text
dist(q, x) = 1 - dot(q, x)
```

The program first tries to load the DEEP100K files from a data directory:

- `DEEP100K.base.100k.fbin`
- `DEEP100K.query.fbin`
- `DEEP100K.gt.query.100k.top100.bin`

If the dataset is unavailable, it generates deterministic synthetic vectors
with the same default scale (`100000 x 96`) and validates GPU results against a
CPU exact reference on a query subset.

## Download DEEP100K

The public DEEP base file is a 384 GB 1B-vector file. The helper script uses an
HTTP Range request to download only the first 100K vectors and rewrites the
`.fbin` header to `100000, 96`.

```bash
cd gpu_ann
bash scripts/download_deep100k.sh /anndata
```

## Build

```bash
cd gpu_ann/src
make
```

## Run

```bash
# Try DEEP100K from /anndata, 2000 queries, 100000 base vectors.
./ann_gpu_cuda

# Explicit arguments:
# ./ann_gpu_cuda DATA_DIR QUERY_COUNT BASE_LIMIT CHECK_QUERIES
./ann_gpu_cuda /anndata 2000 100000 100
```

The program prints recall/correctness and average GPU latency per query.
