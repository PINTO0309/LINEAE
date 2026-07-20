import re

import torch
import torch.nn as nn


def get_optim_params(cfg: list, model: nn.Module, include_frozen_backbone: bool = False):
    """
    E.g.:
        ^(?=.*a)(?=.*b).*$  means including a and b
        ^(?=.*(?:a|b)).*$   means including a or b
        ^(?=.*a)(?!.*b).*$  means including a, but not b
    """

    def is_eligible(name, parameter):
        return parameter.requires_grad or (
            include_frozen_backbone and name.startswith('backbone.')
        )

    eligible_params = {
        name: parameter
        for name, parameter in model.named_parameters()
        if is_eligible(name, parameter)
    }
    param_groups = []
    visited = []
    for pg_config in cfg:
        pattern = pg_config['params']
        params = {
            name: parameter
            for name, parameter in eligible_params.items()
            if len(re.findall(pattern, name)) > 0
        }
        group = {key: value for key, value in pg_config.items() if key != 'params'}
        group['params'] = params.values()
        param_groups.append(group)
        visited.extend(list(params.keys()))

    names = list(eligible_params)

    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {
            name: parameter
            for name, parameter in eligible_params.items()
            if name in unseen
        }
        param_groups.append({'params': params.values()})
        visited.extend(list(params.keys()))

    assert len(visited) == len(names), ''

    return param_groups


def build_adamw_optimizer(args, model: nn.Module, device: torch.device):
    """Build the one authoritative LINEAE AdamW, fused only on CUDA."""
    param_groups = get_optim_params(
        args.model_parameters,
        model,
        include_frozen_backbone=getattr(args, "progressive_unfreeze", False),
    )
    fused = bool(getattr(args, "optimizer_fused", False) and device.type == "cuda")
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
        fused=fused,
    )
    optimizer.lineae_fused = fused
    return optimizer
