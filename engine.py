# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
import util.misc as utils
from util.training_state import atomic_torch_save


def _parameter_group_grad_norms(optimizer):
    norms = []
    for group in optimizer.param_groups:
        gradients = [
            parameter.grad.detach().float().norm(2)
            for parameter in group['params']
            if parameter.grad is not None
        ]
        if gradients:
            norms.append(float(torch.stack(gradients).norm(2)))
        else:
            norms.append(0.0)
    return norms


def _move_batch_to_device(samples, targets, device, args):
    non_blocking = bool(
        device.type == 'cuda' and getattr(args, 'pin_memory', True)
    )
    samples = samples.to(device, non_blocking=non_blocking)
    targets = [
        {key: value.to(device, non_blocking=non_blocking) for key, value in target.items()}
        for target in targets
    ]
    return samples, targets


class _StepPhaseProfiler:
    def __init__(self, device):
        self.device = device
        self.timings = {}
        self.started = {}
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            self.overall_started = torch.cuda.Event(enable_timing=True)
            self.overall_started.record(torch.cuda.current_stream(device))
        else:
            self.overall_started = time.perf_counter()

    def start(self, name):
        if self.device.type == 'cuda':
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.started[name] = event
        else:
            self.started[name] = time.perf_counter()

    def stop(self, name):
        started = self.started.pop(name)
        if self.device.type == 'cuda':
            stopped = torch.cuda.Event(enable_timing=True)
            stopped.record(torch.cuda.current_stream(self.device))
            self.timings[name] = (started, stopped)
        else:
            self.timings[name] = (time.perf_counter() - started) * 1000.0

    def finish(self, *, batch_size, input_size, optimizer_stepped):
        if self.device.type == 'cuda':
            overall_stopped = torch.cuda.Event(enable_timing=True)
            overall_stopped.record(torch.cuda.current_stream(self.device))
            torch.cuda.synchronize(self.device)
            timings = {
                name: float(started.elapsed_time(stopped))
                for name, (started, stopped) in self.timings.items()
            }
            step_ms = float(self.overall_started.elapsed_time(overall_stopped))
        else:
            timings = self.timings
            step_ms = (time.perf_counter() - self.overall_started) * 1000.0
        peak_memory_mib = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            if self.device.type == 'cuda'
            else None
        )
        phases = {
            name: timings.get(name, 0.0)
            for name in (
                'transfer_ms',
                'student_supervised_ms',
                'teacher_forward_ms',
                'kd_loss_ms',
                'backward_ms',
                'optimizer_ms',
            )
        }
        return {
            **phases,
            'online_kd_ms': phases['teacher_forward_ms'] + phases['kd_loss_ms'],
            'step_ms': step_ms,
            'throughput_images_per_second': batch_size * 1000.0 / step_ms,
            'peak_memory_mib': peak_memory_mib,
            'batch_size': batch_size,
            'input_size': input_size,
            'optimizer_stepped': optimizer_stepped,
        }


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer=None,
                    lr_scheduler=None, warmup_scheduler=None, args=None, ema_m=None,
                    max_steps=None, scaler=None, start_global_step=0,
                    teacher_model=None, distillation_criterion=None,
                    step_profile_callback=None):
    if scaler is None:
        scaler = torch.amp.GradScaler(
            str(device),
            enabled=args.amp,
            init_scale=getattr(args, 'amp_init_scale', 65536.0),
        )
    model.train()
    criterion.train()
    if (teacher_model is None) != (distillation_criterion is None):
        raise ValueError('teacher_model and distillation_criterion must be provided together')
    if teacher_model is not None:
        teacher_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    steps_completed = 0
    batches_processed = 0
    accumulation_steps = max(1, int(getattr(args, 'gradient_accumulation_steps', 1)))
    loader_length = len(data_loader)
    optimizer.zero_grad(set_to_none=True)

    tracked_before = {}
    if getattr(args, 'verify_optimizer_step', False):
        wanted_prefixes = ('backbone.pyramid', 'backbone.p5', 'encoder', 'decoder')
        found_prefixes = set()
        for name, parameter in model.named_parameters():
            normalized_name = name.removeprefix('module.')
            for prefix in wanted_prefixes:
                if (prefix not in found_prefixes and normalized_name.startswith(prefix)
                        and parameter.requires_grad):
                    tracked_before[normalized_name] = parameter.detach().clone()
                    found_prefixes.add(prefix)
                    break

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batches_processed += 1
        step_profiler = _StepPhaseProfiler(device) if step_profile_callback is not None else None
        if step_profiler is not None:
            step_profiler.start('transfer_ms')
        samples, targets = _move_batch_to_device(samples, targets, device, args)
        if step_profiler is not None:
            step_profiler.stop('transfer_ms')
        global_step = start_global_step + steps_completed
        window_start = (i // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, loader_length - window_start)
        should_step = (i - window_start + 1) == window_size
        sync_context = (
            model.no_sync() if hasattr(model, 'no_sync') and not should_step else nullcontext()
        )

        with sync_context:
            kd_schedule = None
            kd_match_count = None
            kd_candidate_count = None
            kd_rejected_count = None
            kd_target_coverage = None
            kd_mean_confidence = None
            kd_match_weight_sum = None
            kd_overhead_ms = None
            with torch.amp.autocast(str(device), enabled=args.amp):
                if step_profiler is not None:
                    step_profiler.start('student_supervised_ms')
                outputs = model(samples, targets)
                loss_dict = criterion(outputs, targets)
                if step_profiler is not None:
                    step_profiler.stop('student_supervised_ms')
                if teacher_model is not None:
                    kd_started = time.perf_counter()
                    if step_profiler is not None:
                        step_profiler.start('teacher_forward_ms')
                    with torch.no_grad():
                        teacher_outputs = teacher_model(samples, None)
                    if step_profiler is not None:
                        step_profiler.stop('teacher_forward_ms')
                        step_profiler.start('kd_loss_ms')
                    kd_losses = distillation_criterion(
                        outputs,
                        teacher_outputs,
                        global_step=global_step,
                        targets=targets,
                    )
                    if step_profiler is not None:
                        step_profiler.stop('kd_loss_ms')
                    kd_schedule = distillation_criterion.schedule(global_step)
                    kd_match_count = distillation_criterion.last_match_count
                    kd_candidate_count = (
                        distillation_criterion.last_teacher_candidate_count
                    )
                    kd_rejected_count = (
                        distillation_criterion.last_teacher_rejected_count
                    )
                    kd_target_coverage = (
                        distillation_criterion.last_target_coverage
                    )
                    kd_mean_confidence = (
                        distillation_criterion.last_mean_confidence
                    )
                    kd_match_weight_sum = (
                        distillation_criterion.last_match_weight_sum
                    )
                    # Hungarian matching batches all non-empty image costs into
                    # one CPU transfer, which synchronizes preceding CUDA work and
                    # makes this wall time a meaningful online-KD measurement.
                    kd_overhead_ms = (
                        None
                        if step_profiler is not None
                        else (time.perf_counter() - kd_started) * 1000.0
                    )
                    loss_dict.update(kd_losses)
                losses = sum(loss_dict.values())

            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values()).item()
            if not math.isfinite(loss_value):
                if utils.is_main_process() and getattr(args, 'output_dir', None):
                    diagnostic_path = Path(args.output_dir) / f'nonfinite_step_{global_step}.pth'
                    atomic_torch_save({
                        'model': (model.module if hasattr(model, 'module') else model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'losses': {
                            key: value.detach().cpu() for key, value in loss_dict_reduced.items()
                        },
                    }, diagnostic_path)
                raise FloatingPointError(
                    f'non-finite loss at step {global_step}: {loss_dict_reduced}'
                )

            backward_loss = losses / window_size
            if step_profiler is not None:
                step_profiler.start('backward_ms')
            if args.amp:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            if step_profiler is not None:
                step_profiler.stop('backward_ms')

        grad_norm = None
        group_grad_norms = None
        optimizer_stepped = False
        if step_profiler is not None:
            step_profiler.start('optimizer_ms')
        if should_step:
            if args.amp:
                scaler.unscale_(optimizer)
            group_grad_norms = _parameter_group_grad_norms(optimizer)
            if max_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if args.amp:
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = float(scaler.get_scale()) >= scale_before
            else:
                optimizer.step()
                optimizer_stepped = True

            if tracked_before:
                if not optimizer_stepped:
                    raise RuntimeError(
                        'AMP overflow skipped the bounded smoke optimizer step'
                    )
                named_parameters = {
                    name.removeprefix('module.'): parameter
                    for name, parameter in model.named_parameters()
                }
                changed = [
                    name for name, before in tracked_before.items()
                    if not torch.equal(before, named_parameters[name].detach())
                ]
                print(f'Optimizer-step changed tracked parameters: {changed}')
                if not changed:
                    raise RuntimeError(
                        'optimizer step changed no tracked parameters; check AMP overflow and learning rates'
                    )
                tracked_before.clear()

            optimizer.zero_grad(set_to_none=True)
            if optimizer_stepped:
                if warmup_scheduler is not None:
                    warmup_scheduler.step()
                if (
                    lr_scheduler is not None
                    and getattr(args, 'scheduler_step_unit', 'epoch') == 'optimizer'
                    and (warmup_scheduler is None or warmup_scheduler.finished())
                ):
                    lr_scheduler.step()
                if args.use_ema and epoch >= args.ema_epoch:
                    if ema_m is None:
                        raise RuntimeError('use_ema=True but no EMA model was provided')
                    ema_m.update(model)
                steps_completed += 1
        if step_profiler is not None:
            step_profiler.stop('optimizer_ms')
            profile = step_profiler.finish(
                batch_size=int(samples.shape[0]),
                input_size=list(samples.shape[-2:]),
                optimizer_stepped=optimizer_stepped,
            )
            profile.update({
                'global_step': global_step,
                'loss': loss_value,
                'kd_matches': kd_match_count,
                'kd_candidates': kd_candidate_count,
                'kd_rejected': kd_rejected_count,
                'kd_target_coverage': kd_target_coverage,
                'kd_mean_confidence': kd_mean_confidence,
                'kd_match_weight_sum': kd_match_weight_sum,
                'kd_weight': kd_schedule.weight if kd_schedule is not None else 0.0,
                'kd_temperature': (
                    kd_schedule.temperature if kd_schedule is not None else None
                ),
            })
            if kd_schedule is not None:
                kd_overhead_ms = profile['online_kd_ms']
            step_profile_callback(profile)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if grad_norm is not None:
            metric_logger.update(grad_norm=float(grad_norm))
        if group_grad_norms is not None:
            for group_index, group_norm in enumerate(group_grad_norms):
                metric_logger.update(**{f'grad_norm_pg_{group_index}': group_norm})
        if should_step and args.amp:
            metric_logger.update(amp_overflow=0.0 if optimizer_stepped else 1.0)
        if kd_schedule is not None:
            metric_logger.update(
                kd_weight=kd_schedule.weight,
                kd_temperature=kd_schedule.temperature,
                kd_matches=kd_match_count,
                kd_candidates=kd_candidate_count,
                kd_rejected=kd_rejected_count,
                kd_target_coverage=kd_target_coverage,
                kd_mean_confidence=kd_mean_confidence,
                kd_match_weight_sum=kd_match_weight_sum,
                kd_overhead_ms=kd_overhead_ms,
            )

        if i == 0 and utils.is_main_process():
            underlying = model.module if hasattr(model, 'module') else model
            feature_shapes = getattr(getattr(underlying, 'backbone', None), 'last_feature_shapes', None)
            print(f'Backbone feature shapes: {feature_shapes}')
            print(f'Output shapes: pred_logits={tuple(outputs["pred_logits"].shape)}, '
                  f'pred_lines={tuple(outputs["pred_lines"].shape)}')

        optimizer_global_step = start_global_step + steps_completed
        if (
            should_step
            and optimizer_stepped
            and writer
            and utils.is_main_process()
            and optimizer_global_step % 10 == 0
        ):
            writer.add_scalar('Loss/total', loss_value, optimizer_global_step)
            for group_index, group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{group_index}', group['lr'], optimizer_global_step)
                writer.add_scalar(
                    f'GradNorm/pg_{group_index}', group_grad_norms[group_index], optimizer_global_step
                )
            for key, value in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{key}', value.item(), optimizer_global_step)
            if kd_schedule is not None:
                writer.add_scalar('Distillation/weight', kd_schedule.weight, optimizer_global_step)
                writer.add_scalar(
                    'Distillation/temperature', kd_schedule.temperature, optimizer_global_step
                )
                writer.add_scalar('Distillation/matches', kd_match_count, optimizer_global_step)
                writer.add_scalar(
                    'Distillation/candidates', kd_candidate_count, optimizer_global_step
                )
                writer.add_scalar(
                    'Distillation/rejected', kd_rejected_count, optimizer_global_step
                )
                writer.add_scalar(
                    'Distillation/target_coverage',
                    kd_target_coverage,
                    optimizer_global_step,
                )
                writer.add_scalar(
                    'Distillation/mean_confidence',
                    kd_mean_confidence,
                    optimizer_global_step,
                )
                writer.add_scalar(
                    'Distillation/match_weight_sum',
                    kd_match_weight_sum,
                    optimizer_global_step,
                )
                writer.add_scalar(
                    'Distillation/overhead_ms', kd_overhead_ms, optimizer_global_step
                )

        if should_step and max_steps is not None and steps_completed >= max_steps:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items() if meter.count > 0}
    epoch_complete = batches_processed == loader_length
    return stats, steps_completed, epoch_complete


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, args=None):
    from datasets import DualLineEvaluator

    model.eval()
    criterion.eval()
    evaluator = DualLineEvaluator(deploy_max_predictions=args.num_select)

    metric_logger = utils.MetricLogger(delimiter="  ")
   
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 250, header):
        samples, targets = _move_batch_to_device(samples, targets, device, args)

        with torch.amp.autocast(str(device), enabled=args.amp):
            outputs = model(samples, targets)

            loss_dict = criterion(outputs, targets)
            evaluator.update(outputs, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        metric_logger.update(loss=sum(loss_dict_reduced.values()),
                             **loss_dict_reduced,)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    print("Averaged stats:", metric_logger)
        
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stats.update(evaluator.sap_results)

    return stats

@torch.no_grad()
def test(model, criterion, postprocessors, evaluator, data_loader, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    evaluator.cleanup()
   
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 250, header):
        samples, targets = _move_batch_to_device(samples, targets, device, args)

        outputs = model(samples, targets)

        evaluator.update(outputs, targets)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluator.accumulate()
    evaluator.summarize()

            
    return dict(evaluator.sap_results)
