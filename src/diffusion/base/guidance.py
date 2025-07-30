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

def c4_guidance_fn(out, cfg):
    # guidance function in DiT/SiT, seems like a bug not a feature?
    uncondition, condition = out.chunk(2, dim=0)
    out = condition
    out[:, :4] = uncondition[:, :4] + cfg * (condition[:, :4] - uncondition[:, :4])
    out[:, 4:] = uncondition[:, 4:] + 1.05 * (condition[:, 4:] - uncondition[:, 4:])
    return out

def c4_p05_guidance_fn(out, cfg):
    # guidance function in DiT/SiT, seems like a bug not a feature?
    uncondition, condition = out.chunk(2, dim=0)
    out = condition
    out[:, :4] = uncondition[:, :4] + cfg * (condition[:, :4] - uncondition[:, :4])
    out[:, 4:] = uncondition[:, 4:] + 1.05 * (condition[:, 4:] - uncondition[:, 4:])
    return out

def c4_p10_guidance_fn(out, cfg):
    # guidance function in DiT/SiT, seems like a bug not a feature?
    uncondition, condition = out.chunk(2, dim=0)
    out = condition
    out[:, :4] = uncondition[:, :4] + cfg * (condition[:, :4] - uncondition[:, :4])
    out[:, 4:] = uncondition[:, 4:] + 1.10 * (condition[:, 4:] - uncondition[:, 4:])
    return out

def c4_p15_guidance_fn(out, cfg):
    # guidance function in DiT/SiT, seems like a bug not a feature?
    uncondition, condition = out.chunk(2, dim=0)
    out = condition
    out[:, :4] = uncondition[:, :4] + cfg * (condition[:, :4] - uncondition[:, :4])
    out[:, 4:] = uncondition[:, 4:] + 1.15 * (condition[:, 4:] - uncondition[:, 4:])
    return out

def c4_p20_guidance_fn(out, cfg):
    # guidance function in DiT/SiT, seems like a bug not a feature?
    uncondition, condition = out.chunk(2, dim=0)
    out = condition
    out[:, :4] = uncondition[:, :4] + cfg * (condition[:, :4] - uncondition[:, :4])
    out[:, 4:] = uncondition[:, 4:] + 1.20 * (condition[:, 4:] - uncondition[:, 4:])
    return out

def p4_guidance_fn(out, cfg):
    # guidance function in DiT/SiT, seems like a bug not a feature?
    uncondition, condtion = out.chunk(2, dim=0)
    out = condtion
    out[:, 4:] = uncondition[:, 4:] + cfg * (condtion[:, 4:] - uncondition[:, 4:])
    return out
