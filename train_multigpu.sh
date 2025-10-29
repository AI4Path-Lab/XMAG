# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

#!/bin/bash
#SBATCH --account PAS3015
#SBATCH --job-name distill_multigpu
#SBATCH --time=100:00:00
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --output=./log/distill_multigpu_%A_%a.out
#SBATCH --error=./log/distill_multigpu_%A_%a.err
#SBATCH --cluster=ascend

# Activate environment
source activate distill

# Clear conflicting variables
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK

# Basic distributed training settings
export OMP_NUM_THREADS=8
export PYTHONFAULTHANDLER=1

# ===== NCCL 网络修复设置 =====
# 禁用可能有问题的通信方式
export NCCL_IB_DISABLE=1              # 禁用InfiniBand
export NCCL_P2P_DISABLE=1             # 禁用P2P通信
export NCCL_SHM_DISABLE=1             # 禁用共享内存

# 强制使用以太网
export NCCL_NET=Socket                # 强制Socket网络
export NCCL_SOCKET_IFNAME=^lo,^docker,^virbr,^vmnet,^ib  # 排除问题接口

# 超时设置（减少等待时间）
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=9000               # 150分钟超时

# 网络性能调优
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_NET_GDR_LEVEL=0

# 调试信息（问题解决后可以删除）
export NCCL_DEBUG=WARN                # 减少调试输出
export NCCL_DEBUG_SUBSYS=NET          # 只显示网络相关调试

# ===== PyTorch 分布式设置 =====
export MASTER_ADDR="$(scontrol show hostname $SLURM_NODELIST | head -n1).ten.osc.edu"
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE

# 添加PyTorch超时设置
export TORCH_DISTRIBUTED_TIMEOUT=9000  # 150分钟

echo "=== 网络连接测试 ==="
echo "Master node: $MASTER_ADDR"
echo "Testing network connectivity..."


OTHER_NODES=($(scontrol show hostname $SLURM_NODELIST | tail -n +2))
for node in "${OTHER_NODES[@]}"; do
    echo "Testing connection to ${node}.ten.osc.edu..."
    timeout 10 nc -z "${node}.ten.osc.edu" 22 && echo "✓ SSH reachable" || echo "✗ SSH not reachable"
done

echo "Number of nodes: $SLURM_NNODES"
echo "========================="


srun --nodes=$SLURM_NNODES --exclusive \
    python train.py \
        --config configs/pretrain_mg.yaml \
        --output_dir outputs/distill_uni_to_vit_base_pretrain_48x24bsize \
        --run_name "distill_uni_to_vit_base_pretrain_48x24bsize" \
        --tags "multigpu,univ2,48x24bsize,cosineloss"