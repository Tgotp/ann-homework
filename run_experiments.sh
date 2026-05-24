#!/bin/sh

set -u

RESULT_DIR="experiment_results"
SUMMARY="$RESULT_DIR/summary.tsv"
CONFIG_FILE="files/ann_config.txt"

mkdir -p "$RESULT_DIR"

if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$RESULT_DIR/ann_config.backup.txt"
fi

printf "name\tmode\tthreads\tschedule\tchunk\ttop_p\tpq_m\tpq_ks\tpq_iters\tpq_train\trecall\tlatency_us\n" > "$SUMMARY"

write_config() {
    mode="$1"
    threads="$2"
    schedule="$3"
    chunk="$4"
    top_p="$5"
    pq_m="$6"
    pq_ks="$7"
    pq_iters="$8"
    pq_train="$9"

    {
        printf "# mode: 0=Flat baseline, 1=Flat SIMD, 2=Flat OpenMP, 3=Flat OpenMP + SIMD, 4=SQ-SIMD, 5=PQ-SIMD, 6=PQ-SIMD + OpenMP\n"
        printf "mode=%s\n" "$mode"
        printf "threads=%s\n" "$threads"
        printf "schedule=%s\n" "$schedule"
        printf "chunk=%s\n" "$chunk"
        printf "top_p=%s\n" "$top_p"
        printf "pq_m=%s\n" "$pq_m"
        printf "pq_ks=%s\n" "$pq_ks"
        printf "pq_iters=%s\n" "$pq_iters"
        printf "pq_train=%s\n" "$pq_train"
    } > "$CONFIG_FILE"
}

run_case() {
    name="$1"
    mode="$2"
    threads="$3"
    schedule="$4"
    chunk="$5"
    top_p="$6"
    pq_m="$7"
    pq_ks="$8"
    pq_iters="$9"
    pq_train="${10}"

    echo "===== Running $name ====="
    write_config "$mode" "$threads" "$schedule" "$chunk" "$top_p" "$pq_m" "$pq_ks" "$pq_iters" "$pq_train"

    case_dir="$RESULT_DIR/$name"
    mkdir -p "$case_dir"
    cp "$CONFIG_FILE" "$case_dir/ann_config.txt"

    bash test.sh 1 1 > "$case_dir/console.log" 2>&1

    if [ -f test.o ]; then
        cp test.o "$case_dir/test.o"
    fi
    if [ -f test.e ]; then
        cp test.e "$case_dir/test.e"
    fi

    recall="NA"
    latency="NA"
    if [ -f test.o ]; then
        recall=$(awk -F': ' '/average recall/ {print $2}' test.o | tail -n 1)
        latency=$(awk -F': ' '/average latency/ {print $2}' test.o | tail -n 1)
    fi
    if [ "$recall" = "NA" ] || [ -z "$recall" ]; then
        recall=$(awk -F': ' '/average recall/ {print $2}' "$case_dir/console.log" | tail -n 1)
    fi
    if [ "$latency" = "NA" ] || [ -z "$latency" ]; then
        latency=$(awk -F': ' '/average latency/ {print $2}' "$case_dir/console.log" | tail -n 1)
    fi
    if [ -z "$recall" ]; then
        recall="NA"
    fi
    if [ -z "$latency" ]; then
        latency="NA"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$name" "$mode" "$threads" "$schedule" "$chunk" "$top_p" "$pq_m" "$pq_ks" "$pq_iters" "$pq_train" "$recall" "$latency" >> "$SUMMARY"
}

# Flat baseline and SIMD/OpenMP variants
run_case "flat_baseline_t1" 0 1 static 256 2000 8 128 3 5000
run_case "flat_simd_t1" 1 1 static 256 2000 8 128 3 5000
run_case "flat_openmp_t1" 2 1 static 256 2000 8 128 3 5000
run_case "flat_openmp_t2" 2 2 static 256 2000 8 128 3 5000
run_case "flat_openmp_t4" 2 4 static 256 2000 8 128 3 5000
run_case "flat_openmp_t8" 2 8 static 256 2000 8 128 3 5000
run_case "flat_omp_simd_t1" 3 1 static 256 2000 8 128 3 5000
run_case "flat_omp_simd_t2" 3 2 static 256 2000 8 128 3 5000
run_case "flat_omp_simd_t4" 3 4 static 256 2000 8 128 3 5000
run_case "flat_omp_simd_t8" 3 8 static 256 2000 8 128 3 5000

# SQ-SIMD latency-recall trade-off
run_case "sq_topp1000" 4 1 static 256 1000 8 128 3 5000
run_case "sq_topp2000" 4 1 static 256 2000 8 128 3 5000
run_case "sq_topp5000" 4 1 static 256 5000 8 128 3 5000
run_case "sq_topp10000" 4 1 static 256 10000 8 128 3 5000

# PQ-SIMD latency-recall trade-off. Start with moderate KMeans settings.
run_case "pq_topp2000" 5 1 static 256 2000 8 128 3 5000
run_case "pq_topp5000" 5 1 static 256 5000 8 128 3 5000
run_case "pq_topp10000" 5 1 static 256 10000 8 128 3 5000

# PQ-SIMD + OpenMP
run_case "pq_omp_t1" 6 1 static 256 5000 8 128 3 5000
run_case "pq_omp_t2" 6 2 static 256 5000 8 128 3 5000
run_case "pq_omp_t4" 6 4 static 256 5000 8 128 3 5000
run_case "pq_omp_t8" 6 8 static 256 5000 8 128 3 5000

# Schedule/chunk comparison for the strongest Flat OpenMP+SIMD variant.
run_case "sched_static_256" 3 4 static 256 2000 8 128 3 5000
run_case "sched_dynamic_256" 3 4 dynamic 256 2000 8 128 3 5000
run_case "sched_guided_256" 3 4 guided 256 2000 8 128 3 5000
run_case "sched_static_64" 3 4 static 64 2000 8 128 3 5000
run_case "sched_static_1024" 3 4 static 1024 2000 8 128 3 5000

echo "All experiments finished. Summary: $SUMMARY"
