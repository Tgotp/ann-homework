# ANN 单机并行搜索实验

本仓库为并行程序设计 ANN 作业代码，围绕近似最近邻搜索中的核心距离计算与候选筛选过程，完成了 ARM 平台上的 SIMD 与 OpenMP 多线程优化实验。

实验平台主要为课程服务器的 ARM AArch64 环境。代码在 AArch64 上使用 NEON 指令实现手写 SIMD，同时保留了 x86 SIMD 与标量回退分支，便于在不同平台上编译。

## 实现内容

搜索版本通过 `files/ann_config.txt` 中的 `mode` 参数选择：

| mode | 版本 | 说明 |
| --- | --- | --- |
| 0 | Flat baseline | 原始暴力扫描基线 |
| 1 | Flat SIMD | 使用 NEON 对浮点内积进行 SIMD 优化 |
| 2 | Flat OpenMP | 对 base 向量扫描过程进行 OpenMP 并行 |
| 3 | Flat OpenMP + SIMD | 多线程与 NEON SIMD 结合 |
| 4 | SQ-SIMD | 标量量化，使用 uint8 近似内积筛选候选，再用原始浮点距离重排 |
| 5 | PQ-SIMD | 乘积量化，基于子空间 KMeans 训练码本，使用查表距离筛选候选，再重排 |
| 6 | PQ-SIMD + OpenMP | 在 PQ 查询阶段加入 OpenMP 并行 |

其中 SQ/PQ 版本均支持 `top_p` 候选数量控制，用于观察召回率与查询延迟之间的权衡。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `main.cc` | 作业提供的主程序入口，加入配置读取、索引预处理与搜索分发 |
| `ann_search_versions.h` | 本次实验的主要实现，包括 Flat/SQ/PQ、NEON SIMD、OpenMP 版本 |
| `flat_scan.h` | 原始 Flat baseline |
| `files/ann_config.txt` | 运行时参数配置文件 |
| `run_experiments.sh` | 自动批量运行实验并收集结果 |
| `experiment_results/summary.tsv` | 已完成实验的结果汇总 |
| `test.sh` / `qsub.sh` | 课程服务器提交与测试脚本 |

## 参数说明

`files/ann_config.txt` 示例：

```text
mode=3
threads=4
schedule=static
chunk=256
top_p=2000
pq_m=8
pq_ks=128
pq_iters=3
pq_train=5000
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `mode` | 搜索版本编号 |
| `threads` | OpenMP 线程数，仅对多线程版本有效 |
| `schedule` | OpenMP 调度策略，可取 `static`、`dynamic`、`guided` |
| `chunk` | OpenMP 调度块大小 |
| `top_p` | SQ/PQ 近似筛选后进入精确重排的候选数量 |
| `pq_m` | PQ 子空间数量 |
| `pq_ks` | 每个 PQ 子空间的聚类中心数量，当前实验使用 128 |
| `pq_iters` | PQ 子空间 KMeans 迭代次数 |
| `pq_train` | PQ 训练采样向量数量 |

## 运行方式

在课程服务器 `homework/ann` 目录下，修改 `files/ann_config.txt` 后运行：

```sh
bash test.sh 1 1
```

其中两个 `1` 是课程脚本参数，搜索版本和线程数由 `files/ann_config.txt` 控制。

批量运行全部实验：

```sh
bash run_experiments.sh
```

运行结束后会生成或更新：

```text
experiment_results/summary.tsv
```

## 实验数据

实验使用课程服务器提供的 DEEP100K 数据：

| 数据 | 路径 |
| --- | --- |
| Query | `/anndata/DEEP100K.query.fbin` |
| Ground Truth | `/anndata/DEEP100K.gt.query.100k.top100.bin` |
| Base | `/anndata/DEEP100K.base.100k.fbin` |

当前主程序中查询数量设置为前 2000 条，返回 Top-10 结果，并统计平均召回率与平均查询延迟。

## 实验环境

主要实验环境：

| 项目 | 配置 |
| --- | --- |
| CPU 架构 | AArch64 |
| CPU | HiSilicon Kunpeng-920 |
| 核心数 | 8 |
| SIMD | ARM NEON / ASIMD |
| 编译器 | GCC 10.3.1 |
| 并行框架 | OpenMP |

## 当前结果

完整结果见 `experiment_results/summary.tsv`。

部分代表性结果：

| 版本 | 线程数 | Recall | Latency(us) |
| --- | ---: | ---: | ---: |
| Flat baseline | 1 | 0.99995 | 20851.9 |
| Flat SIMD | 1 | 0.99995 | 7742.58 |
| Flat OpenMP + SIMD | 8 | 0.99995 | 747.756 |
| PQ-SIMD, top_p=2000 | 1 | 0.996301 | 2830.47 |
| PQ-SIMD, top_p=10000 | 1 | 0.99995 | 8019.73 |

实验表明，Flat 搜索中 NEON SIMD 和 OpenMP 能够明显降低查询延迟；PQ-SIMD 在保持较高召回率的同时可以进一步减少计算量；SQ-SIMD 速度较快但召回率损失较明显。
