# 并行程序设计期末研究报告

本目录为期末提交材料草稿：

- `main.tex`：期末研究报告 LaTeX 源文件。
- `references.bib`：参考文献。
- `NKU.png`：封面校徽。
- `experiment_results/gpu_ann_cuda.txt`：CUDA Flat ANNS 运行结果。
- `src/gpu_ann/`：CUDA Flat ANNS 源码快照。
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
```
