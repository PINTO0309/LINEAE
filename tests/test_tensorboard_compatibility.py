from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter


def test_tensorboard_reads_tensorboardx_scalar_events(tmp_path):
    writer = SummaryWriter(logdir=str(tmp_path))
    writer.add_scalar("Train/loss", 1.25, 7)
    writer.close()

    events = EventAccumulator(str(tmp_path))
    events.Reload()

    scalar = events.Scalars("Train/loss")
    assert len(scalar) == 1
    assert scalar[0].step == 7
    assert scalar[0].value == 1.25
