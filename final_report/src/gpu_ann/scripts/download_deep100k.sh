#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/anndata}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

BASE_URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin"
QUERY_URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin"

# fbin header is two little-endian uint32 values: n, dim.
# The public DEEP base file contains 1B vectors. For DEEP100K we only need the
# first 100000 vectors after the 8-byte header:
# 100000 * 96 * sizeof(float) = 38400000 bytes.
if [[ ! -f DEEP100K.base.100k.fbin ]]; then
  echo "Downloading first 100K vectors from DEEP base..."
  curl -L --fail --retry 3 -r 8-38400007 -o DEEP100K.base.100k.payload "${BASE_URL}"
  perl -e 'print pack("II", 100000, 96)' > DEEP100K.base.100k.fbin
  cat DEEP100K.base.100k.payload >> DEEP100K.base.100k.fbin
  rm -f DEEP100K.base.100k.payload
fi

if [[ ! -f DEEP100K.query.fbin ]]; then
  echo "Downloading DEEP public query vectors..."
  curl -L --fail --retry 3 -o DEEP100K.query.fbin "${QUERY_URL}"
fi

ls -lh DEEP100K.base.100k.fbin DEEP100K.query.fbin
perl -e 'for $f (@ARGV) { open(F,"<",$f) or die $!; read(F,$b,8); @x=unpack("II",$b); print "$f n=$x[0] dim=$x[1] bytes=", -s $f, "\n"; close F; }' \
  DEEP100K.base.100k.fbin DEEP100K.query.fbin
