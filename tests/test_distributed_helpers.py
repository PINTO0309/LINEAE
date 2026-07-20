from types import SimpleNamespace

from util.misc import init_distributed_mode


def test_no_torchrun_environment_selects_single_process(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace()

    init_distributed_mode(args)

    assert args.distributed is False
    assert args.rank == 0
    assert args.world_size == 1
    assert args.local_rank == 0
