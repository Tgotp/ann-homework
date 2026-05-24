#mpi.sh
#!/bin/sh # 这 是 脚 本 的 解 释 器 行 ， 表示使用Bash shell 解释脚本
#PBS −N mpi # 任务 程 序为mpi
#PBS −l nodes=4 # 需 要 四 个 节 点 来 计 算

/usr/local/bin/pssh -h $PBS_NODEFILE mkdir -p /home/sTest 1>&2 # 在分 配 到的4 个 计 算 节 点 上 创 建 对 应 的 路 径
scp master_ubss1:/home/sTest/mpi /home/sTest # mpi为 编 译 之 后 的 可 执 行 程 序
/usr/local/bin/pscp -h $PBS_NODEFILE /home/sTest/mpi /home/sTest 1>&2 # 获取master节点中的mpi可执 行程 序 ， 并 分 发 到 每 个 计 算 节 点
mpiexec -np 4 -machinefile $PBS_NODEFILE /home/sTest/mpi # 在4 个 计 算 节 点 上 运 行 可 执 行 程 序mp