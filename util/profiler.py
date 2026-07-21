import copy
from calflops import calculate_flops
import torch
from typing import Tuple

def stats(
    model, args,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    base_size = args.eval_spatial_size[0]
    input_shape = (1, 3, base_size, base_size)

    if getattr(args, "variant", None) in {"2XL", "3XL"}:
        # Reconstruct only the architecture on meta so profiling an 853M
        # parameter model neither duplicates its checkpoint nor executes its
        # multi-teraflop forward on the CPU before the CUDA benchmark starts.
        from models import build_lineae

        meta_args = copy.deepcopy(args)
        meta_args.pretrained = False
        with torch.device("meta"):
            model_for_info, _ = build_lineae(meta_args)
            model_for_info.deploy()
    else:
        model_for_info = copy.deepcopy(model).deploy()

    flops, macs, _ = calculate_flops(model=model_for_info,
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=False)
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info
    return {'flops': flops, 'macs': macs, 'params': params}
