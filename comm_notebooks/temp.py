#!/usr/bin/env python3
"""
group_broadcast_nccl.py

用途：
- 单机 8 张 GPU（cuda:0..7）
- 把 8 张卡分成两个 group，每组 4 张卡：
    group0 = ranks [0,1,2,3], root = 0
    group1 = ranks [4,5,6,7], root = 4
- 在每个 group 内对一个 tensor 做 broadcast（src = group_root）
- 使用 NCCL 后端

运行（单机）：
    python group_broadcast_nccl.py

如果需要自定义 master addr/port：
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 python group_broadcast_nccl.py
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from shutil import rmtree

WORLD_SIZE = 8
GROUP_SIZE = 4

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    # 使用 tcp init method
    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend='hccl',
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size, master_addr, master_port):
    # 每个进程绑定对应 GPU（假设单机）
    torch.npu.set_device(rank)
    device = torch.device(f'npu:{rank}')

    setup(rank, world_size, master_addr, master_port)

    # define groups: two groups of 4
    if rank < GROUP_SIZE:
        group_ranks = list(range(0, GROUP_SIZE))
        group_root = 0
    else:
        group_ranks = list(range(GROUP_SIZE, 2 * GROUP_SIZE))
        group_root = GROUP_SIZE  # 4

    # 创建 subgroup
    pg = dist.new_group(ranks=group_ranks, backend='hccl')

    # 每个进程准备一个 tensor：root 将发送其值，其他接收
    # 用 float tensor 做示例
    tensor = torch.zeros(1, device=device)
    if rank == group_root:
        # root 加入可识别的值（方便观察）
        tensor.fill_(100.0 + float(group_root))  # group0 root -> 100, group1 root -> 104
    else:
        tensor.fill_(0.0)

    # 输出广播前的值（同步打印，为了可读性我们在不同进程间加点延迟）
    print(f"[rank {rank}] before broadcast (group {group_ranks} root {group_root}): {tensor.item()}", flush=True)
    # 同步 barrier（组内同步），方便观察输出顺序
    dist.barrier()

    # 在 group 内广播 tensor（src = group_root）
    dist.broadcast(tensor, src=group_root, group=pg)

    # 再 barrier，确保广播完成
    dist.barrier()

    print(f"[rank {rank}] after  broadcast (should equal root {group_root}): {tensor.item()}", flush=True)

    # 清理 subgroup（可选）
    # 注意: new_group 返回的是 process group object；直接销毁顶层进程组即可
    cleanup()

def main():
    log_dir="/root/ascend"
    if os.path.exists(log_dir):
        rmtree(log_dir)
    # 读取 master 地址/端口（可通过环境变量覆盖）
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = int(os.environ.get('MASTER_PORT', '29500'))
    world_size = WORLD_SIZE
    ngpus = torch.npu.device_count()
    if ngpus < world_size:
        raise ValueError(f"Need {world_size} GPUs but found {ngpus}.")

    # mp.set_start_method("spawn")
    # spawn world_size 个进程
    mp.spawn(
        worker,
        args=(world_size, master_addr, master_port),
        nprocs=world_size,
        join=False
    )

if __name__ == '__main__':
    main()
