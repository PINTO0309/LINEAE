# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path
import os
from types import SimpleNamespace
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import build_adamw_optimizer
from util.artifact_validation import validate_evaluation_report
from util.experiment import config_fingerprint, sha256_file, write_experiment_records
from util.slconfig import DictAction, SLConfig
from util.profiler import stats
from util.model_ema import ModelEMA
from util.training_schedule import build_lr_scheduler, trainable_depth_for_epoch
from util.training_state import (
    atomic_torch_save,
    build_training_checkpoint,
    collect_distributed_rng_states,
    restore_training_checkpoint,
    validate_resume_checkpoint,
)
import util.misc as utils

from datasets import build_dataset, LineEvaluator, BatchImageCollateFunction
from engine import train_one_epoch, evaluate, test
from models.lineae.backbones.base import unwrap_state_dict
from models.lineae.distillation import (
    DistillationTeacher,
    TeacherTargetCache,
    build_distillation_criterion,
    resolve_distillation_temperature_steps,
)
from models.lineae.variants import validate_variant_config

from tensorboardX import SummaryWriter
from warmup import LinearWarmup

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--coco_path', type=str, default='data/wireframe_processed')
    # training parameters
    # parser.add_argument('--output_dir', default='',
    #                     help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--find_unused_params', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--max_train_steps', type=int, default=0,
                        help='stop successfully after this many optimizer steps (0: unlimited)')
    parser.add_argument('--skip_eval', action='store_true',
                        help='skip validation, primarily for bounded smoke runs')
    parser.add_argument('--skip_profile', action='store_true',
                        help='skip FLOP profiling before training')
    parser.add_argument('--verify_optimizer_step', action='store_true',
                        help='fail if a bounded smoke step updates no tracked parameter')

    return parser


def create(args, classname):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    class_module = getattr(args, classname)
    assert class_module in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(class_module)
    return build_func(args)


def data_loader_options(args, device: torch.device) -> dict:
    """Build transfer-oriented DataLoader options without changing sampling."""
    workers = int(args.num_workers)
    if workers < 0:
        raise ValueError('num_workers must be non-negative')
    options = {
        'num_workers': workers,
        'pin_memory': bool(getattr(args, 'pin_memory', True) and device.type == 'cuda'),
    }
    if workers > 0:
        prefetch_factor = int(getattr(args, 'prefetch_factor', 2))
        if prefetch_factor <= 0:
            raise ValueError('prefetch_factor must be positive when workers are enabled')
        options['prefetch_factor'] = prefetch_factor
    return options


def metric_improved(value: float, best: float | None, mode: str) -> bool:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f'non-finite validation selection metric: {value}')
    if mode not in {'max', 'min'}:
        raise ValueError(f"selection_mode must be 'max' or 'min', got {mode!r}")
    if best is None:
        return True
    return value > best if mode == 'max' else value < best


def write_run_completion(
    output_dir: Path,
    *,
    status: str,
    final_epoch: int,
    global_step: int,
    best_metric_name: str,
    best_metric: float | None,
    best_epoch: int | None,
) -> Path:
    if status not in {'full', 'bounded'}:
        raise ValueError(f'unknown completion status: {status!r}')
    checkpoint = output_dir / 'checkpoint.pth'
    best_checkpoint = output_dir / 'checkpoint_best.pth'
    payload = {
        'format': 'lineae_run_completion_v1',
        'status': status,
        'final_epoch': int(final_epoch),
        'global_step': int(global_step),
        'best_metric_name': best_metric_name,
        'best_metric': best_metric,
        'best_epoch': best_epoch,
        'checkpoint_sha256': sha256_file(checkpoint) if checkpoint.is_file() else None,
        'best_checkpoint_sha256': (
            sha256_file(best_checkpoint) if best_checkpoint.is_file() else None
        ),
    }
    path = output_dir / 'run_complete.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return path


def validate_teacher_artifact(checkpoint, config_path: Path) -> None:
    if not isinstance(checkpoint, dict) or checkpoint.get('format') != 'lineae_teacher_v2':
        raise ValueError(
            'distillation requires a qualified lineae_teacher_v2 artifact; '
            'promote a teacher with tools/qualify_teacher.py first'
        )
    expected_path = str(config_path.resolve())
    if checkpoint.get('inference_config') != expected_path:
        raise ValueError(
            'teacher inference config mismatch: '
            f"artifact={checkpoint.get('inference_config')!r}, current={expected_path!r}"
        )
    expected_file_hash = sha256_file(config_path)
    if checkpoint.get('inference_config_file_sha256') != expected_file_hash:
        raise ValueError('teacher inference config file hash mismatch')
    config = SLConfig.fromfile(str(config_path))
    spec = validate_variant_config(config)
    if spec is None:
        raise ValueError('teacher inference config is not a registered LINEAE variant')
    artifact_variant = checkpoint.get('variant')
    if artifact_variant is not None and artifact_variant != spec.name:
        raise ValueError(
            'teacher artifact variant mismatch: '
            f'artifact={artifact_variant!r}, current={spec.name!r}'
        )
    if checkpoint.get('inference_config_sha256') != config_fingerprint(config):
        raise ValueError('teacher inference config hash mismatch')
    model_state = checkpoint.get('model')
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError('teacher artifact has no model state')
    source_hash = checkpoint.get('source_checkpoint_sha256')
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError('teacher artifact has no valid source checkpoint SHA-256')
    source_config_path = Path(checkpoint.get('source_config', ''))
    if not source_config_path.is_file():
        raise ValueError(f'teacher source config is unavailable: {source_config_path}')
    if checkpoint.get('source_config_file_sha256') != sha256_file(source_config_path):
        raise ValueError('teacher source config file hash mismatch')
    source_config = SLConfig.fromfile(str(source_config_path))
    if checkpoint.get('source_config_sha256') != config_fingerprint(source_config):
        raise ValueError('teacher source config hash mismatch')
    baseline_hash = checkpoint.get('baseline_checkpoint_sha256')
    if not isinstance(baseline_hash, str) or len(baseline_hash) != 64:
        raise ValueError('teacher artifact has no valid baseline checkpoint SHA-256')
    baseline_config_path = Path(checkpoint.get('baseline_config', ''))
    if not baseline_config_path.is_file():
        raise ValueError(f'teacher baseline config is unavailable: {baseline_config_path}')
    if checkpoint.get('baseline_config_file_sha256') != sha256_file(baseline_config_path):
        raise ValueError('teacher baseline config file hash mismatch')
    baseline_config = SLConfig.fromfile(str(baseline_config_path))
    if checkpoint.get('baseline_config_sha256') != config_fingerprint(baseline_config):
        raise ValueError('teacher baseline config hash mismatch')
    qualification = checkpoint.get('qualification')
    if not isinstance(qualification, dict):
        raise ValueError('teacher artifact lacks qualification provenance')
    if qualification.get('reload_identical') is not True:
        raise ValueError('teacher artifact did not pass reload identity')
    if qualification.get('canonical_inference_identical') is not True:
        raise ValueError('teacher artifact did not pass canonical inference identity')
    candidate = qualification.get('candidate')
    baseline = qualification.get('baseline')
    if not isinstance(candidate, dict) or not isinstance(baseline, dict):
        raise ValueError('teacher artifact lacks embedded candidate/baseline reports')
    validate_evaluation_report(candidate)
    validate_evaluation_report(baseline)
    if candidate.get('checkpoint_sha256') != source_hash:
        raise ValueError('teacher candidate report does not match its source checkpoint')
    if candidate.get('config') != str(source_config_path.resolve()):
        raise ValueError('teacher candidate report does not match its source config')
    if baseline.get('checkpoint_sha256') != baseline_hash:
        raise ValueError('teacher baseline report does not match its source checkpoint')
    if baseline.get('config') != str(baseline_config_path.resolve()):
        raise ValueError('teacher baseline report does not match its source config')
    minimum_gain = float(qualification.get('minimum_ap10_gain', float('nan')))
    if not math.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError('teacher qualification has an invalid minimum sAP10 gain')
    for dataset in ('wireframe', 'york'):
        candidate_dataset = candidate.get('datasets', {}).get(dataset)
        baseline_dataset = baseline.get('datasets', {}).get(dataset)
        if not isinstance(candidate_dataset, dict) or not isinstance(baseline_dataset, dict):
            raise ValueError(f'teacher qualification lacks {dataset} evaluation')
        for field in ('annotation_sha256', 'samples'):
            if candidate_dataset.get(field) != baseline_dataset.get(field):
                raise ValueError(
                    f'teacher candidate/baseline {dataset}.{field} mismatch'
                )
        annotation_hash = candidate_dataset.get('annotation_sha256')
        samples = candidate_dataset.get('samples')
        if not isinstance(annotation_hash, str) or len(annotation_hash) != 64:
            raise ValueError(f'teacher qualification has invalid {dataset} annotation hash')
        if not isinstance(samples, int) or samples <= 0:
            raise ValueError(f'teacher qualification has invalid {dataset} sample count')
        for metric in ('sap5', 'sap10', 'sap15'):
            for report_name, report_dataset in (
                ('candidate', candidate_dataset),
                ('baseline', baseline_dataset),
            ):
                value = float(report_dataset.get(metric, float('nan')))
                if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise ValueError(
                        f'teacher qualification has invalid {report_name} '
                        f'{dataset}.{metric}'
                    )
    candidate_ap10 = float(candidate['datasets']['wireframe']['sap10'])
    baseline_ap10 = float(baseline['datasets']['wireframe']['sap10'])
    if candidate_ap10 <= baseline_ap10 + minimum_gain:
        raise ValueError('teacher candidate no longer satisfies the recorded sAP10 gate')


def build_frozen_teacher(args, device):
    if getattr(args, 'distill_weight', 0.0) <= 0:
        return None, None
    config_path = Path(args.distill_teacher_config)
    checkpoint_path = Path(args.distill_teacher_checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f'distillation teacher config not found: {config_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'distillation is enabled but teacher checkpoint is missing: {checkpoint_path}'
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not getattr(args, 'distill_allow_unqualified_teacher', False):
        validate_teacher_artifact(checkpoint, config_path)
    elif utils.is_main_process():
        print('WARNING: unqualified distillation teacher explicitly allowed')

    teacher_cfg = SLConfig.fromfile(str(config_path))._cfg_dict.to_dict()
    teacher_args = SimpleNamespace(**teacher_cfg)
    teacher_args.pretrained = False
    teacher_args.return_distill_features = bool(
        getattr(args, 'distill_feature_weight', 0.0) > 0
    )
    # The teacher exposes raw P3/P4/P5 tensors; only the student owns learned
    # projections, so the qualified XL checkpoint remains strictly loadable.
    teacher_args.distill_feature_weight = 0.0
    teacher_model, _ = create(teacher_args, 'modelname')
    state = unwrap_state_dict(checkpoint)
    teacher_model.load_state_dict(state, strict=True)
    teacher_cache = None
    teacher_spatial_size = None
    if getattr(args, 'distill_teacher_resize', True):
        teacher_spatial_size = getattr(teacher_args, 'eval_spatial_size', None)
        if teacher_spatial_size is None:
            raise ValueError('distillation teacher config lacks eval_spatial_size')
        if isinstance(teacher_spatial_size, int):
            teacher_spatial_size = (teacher_spatial_size, teacher_spatial_size)
        else:
            teacher_spatial_size = tuple(teacher_spatial_size)
    cache_dir = getattr(args, 'distill_teacher_cache_dir', '')
    if cache_dir:
        teacher_cache = TeacherTargetCache(
            cache_dir,
            namespace={
                'teacher_checkpoint_sha256': sha256_file(checkpoint_path),
                'teacher_config_sha256': sha256_file(config_path),
                'teacher_resolved_config_sha256': config_fingerprint(
                    SLConfig.fromfile(str(config_path))
                ),
                'source_mean': args.image_mean,
                'source_std': args.image_std,
                'target_mean': teacher_args.image_mean,
                'target_std': teacher_args.image_std,
                'target_spatial_size': teacher_spatial_size,
                'feature_kd': bool(getattr(args, 'distill_feature_weight', 0.0) > 0),
                'amp': bool(args.amp),
                'torch': torch.__version__,
                'device': str(device),
                'gpu': (
                    torch.cuda.get_device_name(device)
                    if torch.cuda.is_available() and torch.device(device).type == 'cuda'
                    else None
                ),
                'cache_schema': 3,
            },
            read_only=getattr(args, 'distill_teacher_cache_read_only', False),
        )
    teacher_model = DistillationTeacher(
        teacher_model,
        source_mean=args.image_mean,
        source_std=args.image_std,
        target_mean=teacher_args.image_mean,
        target_std=teacher_args.image_std,
        target_spatial_size=teacher_spatial_size,
        cache=teacher_cache,
    ).to(device)
    teacher_model.requires_grad_(False).eval()
    criterion = build_distillation_criterion(args).to(device)
    if utils.is_main_process():
        print(
            f'Distillation teacher: config={config_path}, checkpoint={checkpoint_path}, '
            f'sha256={sha256_file(checkpoint_path)}, tensors={len(state)}, '
            f'input={teacher_spatial_size or "student"}'
        )
    return teacher_model, criterion

def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)

    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    backbone_path = Path(args.backbone_weights) if getattr(args, 'backbone_weights', None) else None
    args.backbone_checkpoint_sha256 = (
        sha256_file(backbone_path) if backbone_path is not None and backbone_path.is_file() else None
    )
    teacher_path = (
        Path(args.distill_teacher_checkpoint)
        if getattr(args, 'distill_weight', 0.0) > 0 else None
    )
    args.distill_teacher_checkpoint_sha256 = (
        sha256_file(teacher_path) if teacher_path is not None and teacher_path.is_file() else None
    )
    resume_checkpoint = None
    resume_global_step = 0
    best_metric = None
    best_epoch = None
    selection_metric = getattr(args, 'selection_metric', 'sap10')
    selection_mode = getattr(args, 'selection_mode', 'max').lower()
    if selection_mode not in {'max', 'min'}:
        raise ValueError(f"selection_mode must be 'max' or 'min', got {selection_mode!r}")
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if not args.eval:
            validate_resume_checkpoint(resume_checkpoint, args)
            args.start_epoch = int(resume_checkpoint['epoch']) + 1
            resume_global_step = int(resume_checkpoint['global_step'])
            if resume_checkpoint.get('best_metric_name') not in {None, selection_metric}:
                raise ValueError(
                    'resume checkpoint selection metric mismatch: '
                    f"{resume_checkpoint.get('best_metric_name')!r} != {selection_metric!r}"
                )
            best_metric = resume_checkpoint.get('best_metric')
            best_epoch = resume_checkpoint.get('best_epoch')

    # setup tensorboard writer
    writer = None
    if not args.eval and utils.is_main_process():
        writer_kwargs = {'logdir': args.output_dir}
        if resume_global_step:
            writer_kwargs['purge_step'] = resume_global_step
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(**writer_kwargs)

    if args.eval and args.resume:
        args.pretrained = False

    # setup eval_spatial_size
    if isinstance(args.eval_spatial_size, int):
        size = args.eval_spatial_size 
        args.eval_spatial_size = [size, size]

    assert args.eval_spatial_size[0] == args.eval_spatial_size[1], 'We only support square shapes'
    device = torch.device(args.device)
    loader_options = data_loader_options(args, device)

    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, postprocessors = create(args, 'modelname')
    criterion = create(args, 'criterionname')
    checkpoint_report = getattr(model.backbone, 'checkpoint_report', None)
    if checkpoint_report is not None and utils.is_main_process():
        if hasattr(checkpoint_report, 'loaded_keys'):
            print(
                f'Backbone checkpoint: path={checkpoint_report.path}, '
                f'architecture={checkpoint_report.architecture}, '
                f'loaded={len(checkpoint_report.loaded_keys)}/{checkpoint_report.source_tensor_count}, '
                f'missing={len(checkpoint_report.missing_keys)}, '
                f'unexpected={len(checkpoint_report.unexpected_keys)}, '
                f'shape_mismatches={len(checkpoint_report.shape_mismatches)}, '
                f'strict={checkpoint_report.strict}'
            )
        else:
            print(f'Backbone checkpoint: {checkpoint_report}')
    model.to(device)
    teacher_model, distillation_criterion = build_frozen_teacher(args, device)

    model_without_ddp = model
    total_backbone_blocks = getattr(model_without_ddp.backbone, 'num_blocks', 0)
    desired_trainable_depth = trainable_depth_for_epoch(
        epoch=args.start_epoch,
        total_blocks=total_backbone_blocks,
        initial_depth=getattr(args, 'backbone_trainable_layers', 0),
        initial_freeze_epochs=getattr(args, 'initial_freeze_epochs', 0),
        unfreeze_interval=getattr(args, 'unfreeze_interval', 0),
        progressive=getattr(args, 'progressive_unfreeze', False),
    )
    if args.distributed and hasattr(model_without_ddp.backbone, 'set_trainable_depth'):
        # DDP must see every parameter that may become trainable later.
        model_without_ddp.backbone.set_trainable_depth(0)
    if args.distributed:
        find_unused = args.find_unused_params or getattr(args, 'progressive_unfreeze', False)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=find_unused,
        )
        if find_unused and utils.is_main_process():
            print('DDP find_unused_parameters=True for progressive/frozen backbone parameters')
        model_without_ddp = model.module
    if hasattr(model_without_ddp.backbone, 'set_trainable_depth'):
        model_without_ddp.backbone.set_trainable_depth(desired_trainable_depth)
        print(f'Backbone trainable depth: {desired_trainable_depth}/{total_backbone_blocks}')
    ema_m = None
    if getattr(args, 'use_ema', False):
        ema_m = ModelEMA(model_without_ddp, decay=args.ema_decay, device=device)
        print(
            f'EMA enabled: decay={args.ema_decay}, start_epoch={args.ema_epoch}, '
            f'evaluate={getattr(args, "eval_ema", True)}'
        )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = build_adamw_optimizer(args, model_without_ddp, device)
    print(f'Optimizer: AdamW fused={optimizer.lineae_fused}')
    scaler = torch.amp.GradScaler(
        str(device),
        enabled=args.amp,
        init_scale=getattr(args, 'amp_init_scale', 65536.0),
    )

    if args.eval:
        dataset_val = build_dataset(image_set='val', args=args)
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = DataLoader(
            dataset_val,
            args.batch_size_val,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=BatchImageCollateFunction(base_size=args.eval_spatial_size[0]),
            **loader_options,
        )
    else:
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = None if args.skip_eval else build_dataset(image_set='val', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train, shuffle=True)
            sampler_val = None if dataset_val is None else DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = None if dataset_val is None else torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = DataLoader(dataset_train, 
                                        args.batch_size_train, 
                                        sampler=sampler_train, 
                                        drop_last=True,
                                        collate_fn=BatchImageCollateFunction(
                                            base_size=args.eval_spatial_size[0],
                                            base_size_repeat=3 if getattr(args, 'multi_scale_train', True) else None,
                                            minimum_tokens=(
                                                args.num_queries
                                                if getattr(args, 'multi_scale_train', True)
                                                else None
                                            ),
                                            feature_strides=args.feat_strides,
                                        ),
                                        **loader_options)
        data_loader_val = None
        if dataset_val is not None:
            data_loader_val = DataLoader(dataset_val,
                                            args.batch_size_val,
                                            sampler=sampler_val,
                                            drop_last=False,
                                            collate_fn=BatchImageCollateFunction(base_size=args.eval_spatial_size[0]),
                                            **loader_options)

    if not args.eval:
        args.train_multiscale_scales = list(
            data_loader_train.collate_fn.scales
            or [args.eval_spatial_size[0]]
        )
        accumulation = max(1, int(getattr(args, 'gradient_accumulation_steps', 1)))
        args.optimizer_steps_per_epoch = (len(data_loader_train) + accumulation - 1) // accumulation
    else:
        args.optimizer_steps_per_epoch = 1
    args.distill_temperature_steps_resolved = None
    if distillation_criterion is not None:
        args.distill_temperature_steps_resolved = resolve_distillation_temperature_steps(
            args.distill_temperature_steps,
            optimizer_steps_per_epoch=args.optimizer_steps_per_epoch,
            epochs=args.epochs,
        )
        distillation_criterion.set_temperature_steps(
            args.distill_temperature_steps_resolved
        )
        print(
            'Distillation temperature schedule: '
            f'{args.distill_temperature_start}->{args.distill_temperature_end} over '
            f'{args.distill_temperature_steps_resolved} optimizer-step intervals'
        )
    if resume_checkpoint is not None and not args.eval:
        # The automatic KD horizon depends on the actual per-rank loader length,
        # so validate it after the DataLoader has resolved that value.
        validate_resume_checkpoint(resume_checkpoint, args)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    warmup_scheduler = LinearWarmup(lr_scheduler, args.warmup_iters) if args.use_warmup else None

    output_dir = Path(args.output_dir)

    if resume_checkpoint is not None:
        if args.eval:
            state = unwrap_state_dict(resume_checkpoint)
            model_without_ddp.load_state_dict(state, strict=True)
        else:
            restore_training_checkpoint(
                resume_checkpoint,
                model=model_without_ddp,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                warmup_scheduler=warmup_scheduler,
                scaler=scaler,
                device=device,
                ema_model=ema_m,
            )
            print(f'Resumed full training state from {args.resume} at epoch {args.start_epoch}')

    if args.output_dir and utils.is_main_process():
        write_experiment_records(
            args=args,
            model=model_without_ddp,
            optimizer=optimizer,
            output_dir=output_dir,
            repo_root=Path(__file__).resolve().parent,
        )

    if args.eval:
        evaluation_model = model
        if (
            ema_m is not None
            and getattr(args, 'eval_ema', True)
            and resume_checkpoint is not None
            and resume_checkpoint.get('inference_model') == 'ema_model'
        ):
            ema_m.load_state_dict(resume_checkpoint['ema_model'])
            evaluation_model = ema_m.module
        evaluator = LineEvaluator(max_predictions=args.num_select)
        test_stats = test(evaluation_model, criterion, postprocessors, evaluator,
                        data_loader_val, device, args.output_dir, args=args)
        if utils.is_main_process():
            print(json.dumps(test_stats, sort_keys=True))
        return

    if not args.skip_profile:
        print(stats(model_without_ddp, args))

    print("-"*41 + " Start training " + "-"*42)
    start_time = time.time()
    completed_train_steps = 0
    global_step = resume_global_step
    current_trainable_depth = desired_trainable_depth
    last_epoch = args.start_epoch - 1
    stopped_by_step_bound = False
    for epoch in range(args.start_epoch, args.epochs):
        last_epoch = epoch
        epoch_start_time = time.time()
        target_depth = trainable_depth_for_epoch(
            epoch=epoch,
            total_blocks=total_backbone_blocks,
            initial_depth=getattr(args, 'backbone_trainable_layers', 0),
            initial_freeze_epochs=getattr(args, 'initial_freeze_epochs', 0),
            unfreeze_interval=getattr(args, 'unfreeze_interval', 0),
            progressive=getattr(args, 'progressive_unfreeze', False),
        )
        if target_depth != current_trainable_depth and hasattr(model_without_ddp.backbone, 'set_trainable_depth'):
            model_without_ddp.backbone.set_trainable_depth(target_depth)
            current_trainable_depth = target_depth
            print(f'Progressive unfreeze: {current_trainable_depth}/{total_backbone_blocks} blocks trainable')
        if args.distributed:
            sampler_train.set_epoch(epoch)
        remaining_steps = None
        if args.max_train_steps > 0:
            remaining_steps = args.max_train_steps - completed_train_steps
        train_stats, epoch_steps, epoch_complete = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, lr_scheduler=lr_scheduler, warmup_scheduler=warmup_scheduler, 
            writer=writer, args=args, max_steps=remaining_steps, scaler=scaler,
            start_global_step=global_step, teacher_model=teacher_model,
            distillation_criterion=distillation_criterion, ema_m=ema_m)
        if teacher_model is not None and teacher_model.cache_stats is not None:
            print(f'Distillation teacher cache: {teacher_model.cache_stats}')
        completed_train_steps += epoch_steps
        global_step += epoch_steps
        if getattr(args, 'scheduler_step_unit', 'epoch') == 'epoch':
            if warmup_scheduler is None or warmup_scheduler.finished():
                lr_scheduler.step()
            else:
                print(warmup_scheduler.last_step)

        # eval
        test_stats = {}
        if not args.skip_eval:
            evaluation_model = model
            ema_active = (
                ema_m is not None
                and getattr(args, 'eval_ema', True)
                and epoch >= args.ema_epoch
                and ema_m.num_updates > 0
            )
            if ema_active:
                evaluation_model = ema_m.module
            test_stats = evaluate(
                evaluation_model, criterion, postprocessors, data_loader_val, device,
                args.output_dir, args=args
            )

        is_best = False
        if selection_metric in test_stats:
            current_metric = float(test_stats[selection_metric])
            if metric_improved(current_metric, best_metric, selection_mode):
                best_metric = current_metric
                best_epoch = epoch
                is_best = True
                print(
                    f'New best {selection_metric}={best_metric:.4f} at epoch {best_epoch}'
                )

        checkpoint_rng_states = (
            collect_distributed_rng_states() if args.output_dir else None
        )
        if args.output_dir and utils.is_main_process():
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            interval = int(args.save_checkpoint_interval)
            if interval > 0 and (epoch + 1) % interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if is_best:
                checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
            weights = build_training_checkpoint(
                model=model_without_ddp,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                warmup_scheduler=warmup_scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                args=args,
                repo_root=Path(__file__).resolve().parent,
                best_metric_name=selection_metric,
                best_metric=best_metric,
                best_epoch=best_epoch,
                ema_model=ema_m,
                inference_model='ema_model' if (
                    ema_m is not None
                    and getattr(args, 'eval_ema', True)
                    and epoch >= args.ema_epoch
                    and ema_m.num_updates > 0
                ) else 'model',
                rng_state_by_rank=checkpoint_rng_states,
                epoch_complete=epoch_complete,
            )
            for checkpoint_path in checkpoint_paths:
                atomic_torch_save(weights, checkpoint_path)

        if utils.is_main_process() and writer is not None:
            for k in test_stats:
                writer.add_scalar(f'Test/{k}'.format(k), test_stats[k], epoch)
            
        log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
                'selection_metric': selection_metric,
                'best_metric': best_metric,
                'best_epoch': best_epoch,
            }

        log_stats.update({'now_time': str(datetime.datetime.now())})
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if torch.cuda.is_available() and device.type == 'cuda':
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f'Peak CUDA memory: {peak_mb:.1f} MiB')

        if args.max_train_steps > 0 and completed_train_steps >= args.max_train_steps:
            print(f'Reached max_train_steps={args.max_train_steps}; stopping successfully.')
            stopped_by_step_bound = True
            break
                
    if writer is not None:
        writer.close()

    if args.output_dir and utils.is_main_process():
        status = (
            'full'
            if last_epoch >= args.epochs - 1 and not stopped_by_step_bound
            else 'bounded'
        )
        completion_path = write_run_completion(
            output_dir,
            status=status,
            final_epoch=last_epoch,
            global_step=global_step,
            best_metric_name=selection_metric,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        print(f'Run completion record: {completion_path} ({status})')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LINEA training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
