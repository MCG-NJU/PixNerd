import torch

def simple_guidance_fn(out, cfg):
    uncondition, condtion = out.chunk(2, dim=0)
    out = uncondition + cfg * (condtion - uncondition)
    return out

def c3_guidance_fn(out, cfg):
    # guidance function in DiT/SiT, seems like a bug not a feature?
    uncondition, condtion = out.chunk(2, dim=0)
    out = condtion
    out[:, :3] = uncondition[:, :3] + cfg * (condtion[:, :3] - uncondition[:, :3])
    return out