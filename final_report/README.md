# 并行程序设计期末研究报告

本目录为期末研究报告与复现实验材料：

- `main.tex`：期末研究报告 LaTeX 源文件。
- `references.bib`：参考文献。
- `NKU.png`：封面校徽。
- `experiment_results/gpu_ann_cuda.txt`：CUDA Flat ANNS 运行结果。
- `experiment_results/gpu_ann_cuda_deep100k.txt`：真实 DEEP100K 上的 CUDA Flat ANNS 结果。
- `experiment_results/advanced_ann_bench.tsv`：IVF-Flat、Residual IVF-PQ 与 HNSW 参数实验结果。
- `src/gpu_ann/`：CUDA Flat ANNS 源码快照。
- `src/advanced_bench/`：IVF-Flat、Residual IVF-PQ 与 HNSW 对比实验源码。
- `src/gpu_bitonic/`：GPU 双调排序实验源码快照。

编译方式：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

GPU 实验复核环境：

```bash
ssh -p 34559 root@connect.nmb1.seetacloud.com
cd /root/ann-homework-final/gpu_ann/src
make clean all
./ann_gpu_cuda /anndata 2000 100000 100

cd /root/ann-homework-final/advanced_bench/src
make clean all
./ann_advanced_bench /anndata 200 100000
```
