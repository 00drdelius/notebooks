import torch.distributed as dist

def get_device():
    assert dist.is_initialized(), "distribution is not initialized"
    rank = dist.get_rank()
    match dist.get_backend().lower():
        case "hccl":
            return f"npu:{rank}"
        case "nccl":
            return f"cuda:{rank}"
        case "gloo":
            return "cpu"
