import os

import shutil
ascend_log_dir="/root/ascend"
if os.path.exists(ascend_log_dir):
    # shutil.rmtree(ascend_log_dir)
    ...

os.environ['ASCEND_RT_VISIBLE_DEVICES']="0,1"
os.environ['ASCEND_LAUNCH_BLOCKING']='1'
# os.environ['HCCL_DIAGNOSE_ENABLE']='1'
# os.environ['HCCL_ENTRY_LOG_ENABLE']='1'
# os.environ['HCCL_DEBUG_CONFIG']='ALG,TASK,RESOURCE,AIV_OPS_EXC'

os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = "1" # auto set 32 workers, here limits 1


import multiprocessing
from multiprocessing.context import ForkProcess
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import Backend

local_world_size=2
dummy_tensor_shape=(4096,5120,)


def single_all_reduce(input_:torch.Tensor, device_group:dist.ProcessGroup):
    print(f"tensor before all reduce: {input_}")
    dist.all_reduce(input_, group=device_group, op=dist.ReduceOp.AVG)
    rank = dist.get_rank()
    print(f"rank:{rank} all_reduce success!")
    print(f"tensor after all reduce:{input_}")


class SimulateGroupCoordinator:
    """
    simulate from vllm.distributed.parallel_state.GroupCoordinator
    """
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        **kwargs
    ):
        self.rank = dist.get_rank()
        self.local_rank = local_rank

        self_device_group = None
        self_cpu_group = None

        for ranks in group_ranks:
            device_group = dist.new_group(
                ranks, backend=torch_distributed_backend) #NOTE create sub group in main group
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group


def init_world_group(
    ranks: list[int], local_rank: int, backend: str
) -> SimulateGroupCoordinator:
    """
    simulate from vllm.distributed.parallel_state.init_world_group
    """
    return SimulateGroupCoordinator(
        group_ranks=[ranks], #NOTE GroupCoordinator only used in Single node, so only passing [ranks]
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=False,
        group_name="world",
    )


def local_rank_worker(
    world_size:int = -1,
    rank:int = -1,
    distributed_init_method: str = "env://",
    local_rank:int = -1,
    backend: str = "hccl",
    timeout:timedelta|None = None,
):
    """
    simulate from:
    vllm_acend.worker.worker.NPUWorker._init_worker_distributed_environment;

    Or more specifically,

    vllm.distributed.parallel_state.init_distributed_environment
    """
    if not dist.is_initialized():
        torch.distributed.init_process_group( # only needs to be initialized once.
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        ranks = list(range(torch.distributed.get_world_size()))
        _TP = init_world_group(ranks, local_rank, backend)

        dummy_input = torch.randn(dummy_tensor_shape, device=f"npu:{rank}")
        single_all_reduce(dummy_input, device_group=_TP.device_group)


def main():
    # ctx = get_mp_context()
    ctx = multiprocessing.get_context("spawn")
    processes:list[ForkProcess] = []
    #NOTE simulate from:
    # vllm.v1.executor.multiproc_executor.MultiprocExecutor._init_executor
    for local_rank in range(local_world_size):
        process_kwargs={
            "world_size": local_world_size,
            "local_rank": local_rank,
            "distributed_init_method": "tcp://127.0.0.1:40901",
            "rank": local_rank,
        }
        proc=ctx.Process(
            target=local_rank_worker,
            kwargs=process_kwargs,
            name=f"VllmWorker-{local_rank}",
            daemon=True
        )
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

if __name__ == '__main__':
    print(f"[MAIN] PID={os.getpid()}", flush=True)