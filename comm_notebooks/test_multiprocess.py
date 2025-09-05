#!/usr/bin/python
import os
import sys

os.environ['PYTORCH_NPU_ALLOC_CONF']="expandable_segments:True"
os.environ['ASCEND_RT_VISIBLE_DEVICES']="0,1,2,3,4,5,6,7"

import torch
from torch_npu import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

# torch_npu.npu.mem_get_info
print("[pid]", os.getpid(), "[device_count] ",torch_npu.npu.device_count())
print("[pid]",os.getpid(),"__name__ == %s" % __name__)

def blocking_p2p_comm(rank,world_size,device:str="cpu",dtype=torch.bfloat16):
    tensor=torch.zeros(3,5,dtype=dtype,device=device)
    if rank==0:
        tensor+=torch.ones([3,5],dtype=dtype,device=device)
        # for global_rank in range(1,world_size):
        dist.send(tensor, dst=1) #XXX send tensor to process:rank 1
    elif rank==1:
        dist.recv(tensor, src=0) #XXX receive tensor from process:rank 0
    else:
        #XXX other ranks
        # dist.recv(tensor, src=0)
        pass
    print('Rank ', rank, ' has data: ', tensor)


def init_process(rank, world_size, fn, device,dtype,backend):
    os.environ['MASTER_ADDR']="127.0.0.1"
    os.environ['MASTER_PORT']="11451"
    dist.init_process_group(
        backend,
        rank=rank, world_size=world_size
    )
    fn(rank, world_size, device, dtype)

#XXX 使用`spawn`启动时必须加`if __name__== '__main__'`，否则会报错：`RuntimeError('context has already been set')`
# 该错误来源： python.multiprocessing.context.DefaultContext.set_start_method
# python 执行的主进程 __name__ 会等于 '__main__' ，而其创建的子进程 __name__ 则== '__mp_main__'
# mp.set_start_method("spawn")只会在 "__main__"执行一次，然后设置全局 context ，然后子进程重新执行整个脚本。
# 而子进程的 __name__ != __main__ ，所以不会执行下面的代码，只会有主进程调用

#XXX 而如果使用`fork`： mp.set_start_method("fork") ，则没有上面的要求。
# 调用 `fork` 不会重新执行整个脚本内容，而是会在被执行时复制整个主进程当前的所有资源（可以认为是整个PCB，包括了内存、python模块、变量、TCB等，全部复制粘贴）
# 直觉可以告诉我们这样启动极快，但是很容易出问题。

#XXX 你可以执行一遍"fork"，会发现本脚本 line:14 line:15 只打印了一次，但是 line:29 有4次打印

#XXX reference1: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
#XXX reference2: https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn

# if __name__ == '__main__':
world_size=4
processes:list[mp.Process]=[]
mp.set_start_method("fork")
run=blocking_p2p_comm
for rank in range(world_size):
    p = mp.Process(target=init_process, args=(rank, world_size, run),
                    kwargs=dict(backend="gloo",device="cpu",dtype=torch.bfloat16,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
        