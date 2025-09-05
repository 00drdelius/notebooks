#!/usr/bin/python
import os
import sys
import datetime

os.environ['PYTORCH_NPU_ALLOC_CONF']="expandable_segments:True"
os.environ['ASCEND_RT_VISIBLE_DEVICES']="0,1,2,3,4,5,6,7"

import torch
from torch_npu import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import get_device

# torch_npu.npu.mem_get_info
print("[pid]", os.getpid(), "[device_count] ",torch_npu.npu.device_count())


def blocking_p2p_comm(dtype):
    "blocking peer to peer communication"
    rank=dist.get_rank()
    device=get_device()
    print(f"setting device: {device}")
    #XXX you cann't set the device like torch.tensor([1],device="npu:0"), which induces 4 processes(4 ranks) have
    # tensors only on npu:0, error will raise:
    # RuntimeError: create_config:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:148
    # HCCL function error: hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, &(comm->hcclComm_)), error code is 1
    tensor=torch.zeros(3,5,dtype=dtype,device=device)
    if rank==0:
        tensor+=torch.ones([3,5],dtype=dtype,device=device)
        # for global_rank in range(1,world_size):
        dist.send(tensor, dst=1) #XXX send tensor to process:rank 1
    elif rank==1:
        #XXX here we explicitly set rank==1 to recv otherwise rank 2 & 3 probably raise `Connection closed by peer` error
        # cuz rank 0 process shutdowns once rank1 receives the data
        dist.recv(tensor, src=0) #XXX receive tensor from process:rank 0
    else:
        #XXX other ranks do not execute receive method
        # dist.recv(tensor, src=0)
        pass
    print(f'[rank{rank}]', tensor)


def streaming_p2p_comm(dtype):
    "async concurrent peer to peer communication"
    rank=dist.get_rank()
    device=get_device()
    tensor=torch.zeros(3,5,dtype=dtype,device=device)
    req=None
    if rank==0:
        tensor+=torch.ones([3,5],dtype=dtype,device=device)
        req = dist.isend(tensor,dst=1)
        print("Rank 0 start sending")
    elif rank==1:
        req=dist.irecv(tensor,src=0)
        print("Rank 1 start receiving")
    else:
        pass
    if req!=None:
        req.wait(timeout=datetime.timedelta(seconds=5.0)) #XXX assure communication took place and 
    print(f'[rank{rank}]', tensor)


def blocking_broadcast(dtype):
    "blocking broadcast"
    
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    device=get_device()

    assert world_size%2==0, "world size cannot be even digit."
    
    group1_ranks=[i for i in range(0,world_size//2)]
    group2_ranks=[i for i in range(world_size//2, world_size)]
    #XXX all ranks in `new_group` is sub group of main group, you may see it as a communication domain
    group1 = dist.new_group(ranks=group1_ranks)
    if rank in group1_ranks:
        if rank==group1_ranks[0]:
            tensor=torch.randint(0,10,(3,5),dtype=dtype,device=device)
        else:
            tensor=torch.empty(*(3,5),dtype=dtype, device=device)
        dist.broadcast(tensor, src=group1_ranks[0],group=group1)
        #XXXS the broadcast op might be launched in a different stream
        # need to synchronize to make sure the tensor is ready
        torch_npu.npu.synchronize()


    group2 = dist.new_group(ranks=group2_ranks)
    if rank in group2_ranks:
        if rank==group2_ranks[0]:
            tensor=torch.randint(0,10,(3,5),dtype=dtype,device=device)
        else:
            tensor=torch.empty(*(3,5),dtype=dtype, device=device)
        dist.broadcast(tensor, src=group2_ranks[0],group=group2)
        torch_npu.npu.synchronize()
    
    print(f'[rank{rank}]', tensor)


def init_process(rank, world_size, fn,dtype,backend):
    #XXX dist.init_process_group is required to be executed in all rank processes
    try:
        dist.init_process_group( #XXX this is the main group
            backend,
            # init_method="tcp://{ip}:{port}".format(ip=os.getenv('MASTER_ADDR'),port=os.getenv("MASTER_PORT")),
            init_method="env://",
            rank=rank, world_size=world_size
        )
        print("Rank %s is initialized" % rank)
        fn(dtype)
    except Exception as exc:
        print('[ERROR RAISED] ',exc)
    finally:
        if dist.is_initialized():
            print("destroy all processs group")
            dist.destroy_process_group()

if __name__ == '__main__':
    from shutil import rmtree
    if os.path.exists("/root/ascend"):
        rmtree("/root/ascend") #XXX clear logs to trace error if error raised
    #XXX MASTER_ADDR && MASTER_PORT will be used when `init_method`=="env://" which is by default
    # also you can setup init_method=="tcp://{ip}:{port}" manually. it functions same as env:// with MASTER_ADDR && MASTER_PORT set
    # refer to https://docs.pytorch.org/docs/stable/distributed.html#tcp-initialization to see more details of tcp initialization
    os.environ['MASTER_ADDR']="127.0.0.1"
    os.environ['MASTER_PORT']="11451"
    world_size=4
    processes:list[mp.Process]=[]
    mp.set_start_method("spawn") #XXX spawn method will reimport this script, as to rerun this script again
    # mp.set_start_method("fork") #XXX fork only copy and paste all PCB && TCB from parent process
    run=blocking_broadcast
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run),
                        kwargs=dict(backend="hccl",dtype=torch.bfloat16,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        