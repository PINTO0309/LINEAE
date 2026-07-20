from types import SimpleNamespace

import torch
from torch import nn

from engine import train_one_epoch


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 1, bias=False)

    def forward(self, samples, targets=None):
        logits = self.projection(samples).unsqueeze(1)
        return {
            "pred_logits": logits,
            "pred_lines": logits.new_zeros((samples.shape[0], 1, 4)),
        }


class TinyCriterion(nn.Module):
    def forward(self, outputs, targets):
        expected = torch.stack([target["value"] for target in targets]).reshape(-1, 1, 1)
        return {"loss_logits": (outputs["pred_logits"] - expected).square().mean()}


def _args(accumulation_steps, scheduler_step_unit="epoch"):
    return SimpleNamespace(
        amp=False,
        gradient_accumulation_steps=accumulation_steps,
        verify_optimizer_step=False,
        output_dir="",
        use_ema=False,
        ema_epoch=0,
        scheduler_step_unit=scheduler_step_unit,
    )


def test_accumulated_microbatches_match_one_full_batch():
    inputs = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0]])
    values = [torch.tensor(0.25), torch.tensor(-0.75)]
    microbatches = [
        (inputs[index:index + 1], [{"value": values[index]}])
        for index in range(2)
    ]
    full_batch = [(inputs, [{"value": value} for value in values])]

    initial = TinyDetector()
    accumulated = TinyDetector()
    full = TinyDetector()
    accumulated.load_state_dict(initial.state_dict())
    full.load_state_dict(initial.state_dict())
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    full_optimizer = torch.optim.SGD(full.parameters(), lr=0.1)

    _, accumulated_steps, accumulated_complete = train_one_epoch(
        accumulated,
        TinyCriterion(),
        microbatches,
        accumulated_optimizer,
        torch.device("cpu"),
        epoch=0,
        args=_args(2),
    )
    _, full_steps, full_complete = train_one_epoch(
        full,
        TinyCriterion(),
        full_batch,
        full_optimizer,
        torch.device("cpu"),
        epoch=0,
        args=_args(1),
    )

    assert accumulated_steps == full_steps == 1
    assert accumulated_complete is True
    assert full_complete is True
    assert torch.allclose(accumulated.projection.weight, full.projection.weight, atol=1e-7)


def test_final_partial_accumulation_window_is_scaled_by_its_actual_size():
    inputs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    microbatches = [
        (inputs[index:index + 1], [{"value": torch.tensor(float(index))}])
        for index in range(3)
    ]
    model = TinyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _, steps, epoch_complete = train_one_epoch(
        model,
        TinyCriterion(),
        microbatches,
        optimizer,
        torch.device("cpu"),
        epoch=0,
        args=_args(2),
    )

    assert steps == 2
    assert epoch_complete is True


def test_optimizer_step_scheduler_ignores_accumulation_microbatches():
    inputs = torch.eye(3)
    microbatches = [
        (inputs[index:index + 1], [{"value": torch.tensor(float(index))}])
        for index in range(3)
    ]
    model = TinyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    _, steps, epoch_complete = train_one_epoch(
        model,
        TinyCriterion(),
        microbatches,
        optimizer,
        torch.device("cpu"),
        epoch=0,
        args=_args(2, scheduler_step_unit="optimizer"),
        lr_scheduler=scheduler,
    )

    assert steps == 2
    assert epoch_complete is True
    assert scheduler.last_epoch == 2
    assert optimizer.param_groups[0]["lr"] == 0.025


class OverflowScaler:
    def __init__(self):
        self.updated = False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        return None

    def update(self):
        self.updated = True

    def get_scale(self):
        return 512.0 if self.updated else 1024.0


def test_amp_overflow_does_not_advance_optimizer_scheduler_or_global_step():
    model = TinyDetector()
    before = model.projection.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    args = _args(1, scheduler_step_unit="optimizer")
    args.amp = True
    stats, steps, epoch_complete = train_one_epoch(
        model,
        TinyCriterion(),
        [(torch.ones(1, 3), [{"value": torch.tensor(0.0)}])],
        optimizer,
        torch.device("cpu"),
        epoch=0,
        args=args,
        lr_scheduler=scheduler,
        scaler=OverflowScaler(),
    )

    assert steps == 0
    assert epoch_complete is True
    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert torch.equal(model.projection.weight, before)
    assert stats["amp_overflow"] == 1.0
