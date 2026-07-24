# [WIP] LINEAE

LINEAE (**LINEA E**nhanced) is an experimental successor to [LINEA](https://github.com/SebastianJanampa/LINEA) aimed at improving both line detection accuracy and inference speed. It keeps the LINEA Wireframe/YorkUrban data and detector semantics, and adds selectable HGNetV2/DINOv3 backbones, progressive unfreezing, reproducible resume, XL-to-smaller line-set distillation, an optional qualified-X teacher cascade, EMA, projected feature KD/intermediate-block fusion, exact-input teacher caching, memory-efficient SDPA for every DINO variant, eval-only versioned RoPE caching, allocation-light decoder broadcasts, bounded multi-scale anchor/position caching, and deployment benchmarks.

https://github.com/user-attachments/assets/008f5a87-0411-4599-b699-45d163121d9c

## Implemented variants

| Variant    | Backbone                            |   Initial input |
| ---------- | ----------------------------------- | --------------: |
| A / F / P  | HGNetV2 Atto / Femto / Pico         | 320 / 416 / 640 |
| N / T      | HGNetV2-B0 / B1                     |             640 |
| S / M      | DINOv3 Tiny / Tiny+                 |             640 |
| L / X / XL | official DINOv3 S/16 / S+/16 / B/16 |             640 |
| 2XL / 3XL  | official DINOv3 L/16 / H+/16         |             640 |

The authoritative mapping, including exact bootstrap filenames, is in `models/lineae/variants.py`.

## Parameter inventory

The following exact counts are produced from each committed default config with pretrained loading disabled and after `model.deploy()`, matching the graph used by the Torch/ONNX/TensorRT benchmarks. `Backbone` means `model.backbone`: for DINO variants it includes the Simple Feature Pyramid (SFP), and for A/F/P it includes the efficient synthetic P5. `Head` is the hybrid encoder plus decoder. Default output-KD adds no student parameters; optional feature-KD projections are training-only and are removed by `deploy()`.

**I am not interested in the performance race among architectures; consequently, I do not use appropriate data when evaluating their performance. Specifically, I combine the Wireframe and YorkUrban datasets—datasets characterized by extremely limited training data and a lack of diversity. Therefore, please note that the evaluation scores from `A` to `3XL` in the table below serve only as a guide to model performance. It is entirely meaningless to use the scores in the table below for performance comparisons with existing implementations presented in past research papers.**

- WF: Wireframe, YU: YorkUrban

| Var | Backbone<br>(M) | Head<br>(M) | Total<br>(M) | GFLOPs | WF<br>AP<sup>5</sup> | 　<br>AP<sup>10</sup> | 　<br>AP<sup>15</sup> | YU<br>AP<sup>5</sup> | 　<br>AP<sup>10</sup> | 　<br>AP<sup>15</sup> |
| :-----: | -----------: | -----------------: | --------: | -----: | -------------: | --------------: | --------------: | -------------: | --------------: | --------------: |
| LINEA-N |          1.8 |                2.0 |       3.9 |   11.5 |58.7|65.0|67.9|27.3|30.5|32.5|
| LINEA-S |          2.2 |                6.2 |       8.4 |   29.4 |58.4|64.7|67.6|28.9|32.6|34.8|
| LINEA-M |          6.0 |                7.3 |      13.3 |   43.4 |59.5|66.3|69.1|30.3|34.5|36.7|
| LINEA-L |          13.5 |              11.5 |      25.0 |   81.5 |61.0|67.9|70.8|30.9|34.9|37.3|
| A       |          0.3 |                1.6 |       1.9 |    2.5 |||||||
| F       |          0.7 |                1.9 |       2.6 |    4.7 |||||||
| P       |          1.0 |                2.0 |       3.0 |   10.8 |||||||
| N       |          1.9 |                2.1 |       3.9 |   11.7 |||||||
| T       |          2.2 |                6.2 |       8.4 |   29.4 |63.97|69.35|71.87|52.00|55.83|58.40|
| S       |          6.0 |                5.9 |      11.9 |   39.2 |62.24|68.74|71.49|53.98|59.75|62.72|
| M       |         10.6 |                6.7 |      17.3 |   55.5 |||||||
| L       |         23.0 |                6.7 |      29.7 |   94.5 |||||||
| X       |         30.1 |                8.1 |      38.2 |  121.2 |||||||
| XL      |         88.4 |                8.1 |      96.5 |  306.3 |||||||
| 2XL     |        311.5 |               60.7 |     372.2 | 1173.6 |||||||
| 3XL     |        853.7 |              106.8 |     960.5 | 3043.2 |72.19|76.80|78.65|70.81|74.50|76.48|

`M` is decimal millions (`1 M = 1,000,000` parameters). GFLOPs are the batch-1 forward-operation count reported by the locked `calflops` implementation after `model.deploy()` at each variant's canonical input size: 320 for A, 416 for F, and 640 for P through 3XL. One multiply-accumulate contributes two FLOPs, with other counted operations added separately. Values are rounded to one decimal place; GFLOPs describe graph complexity rather than measured hardware throughput, and the parameter regression test retains the exact integer counts. The 2XL/3XL graph is reconstructed and executed on meta tensors for this accounting, avoiding parameter duplication and real multi-teraflop CPU computation while retaining the same module hooks and shapes.

### A/F/P/N scaling contract

A, F, P, and N now increase strictly in exact deploy parameter count, MACs, and the reference canonical-input latency. The old dense 3x3 A/F/P synthetic P5 alone contained up to 2.36M parameters, making P larger than N. It has been replaced by a 2x2 average downsample followed by a learned 1x1 channel mixer, BatchNorm, and ReLU. N retains HGNetV2-B0's larger native stage 4. To keep real GPU latency from being dominated by the same detector head at every small size, queries scale as 600/800/1100/1200; A additionally uses two decoder layers while F/P/N use three. All variants retain top-300 deployment output. This preserves the three-level `(stride 8, 16, 32)` feature contract while producing the intended A < F < P < N order.

| Variant | Canonical input | Queries | Decoder layers | Exact parameters | MACs (G) | RTX 3070 AMP p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 320 | 600 | 2 | 1,897,494 | 1.2191 | 14.55 |
| F | 416 | 800 | 3 | 2,599,613 | 2.3328 | 16.92 |
| P | 640 | 1,100 | 3 | 2,997,383 | 5.3493 | 20.00 |
| N | 640 | 1,200 | 3 | 3,913,745 | 5.8026 | 20.75 |

The latency snapshot uses the fused deploy graph, batch 1, `--amp`, top-300 selection, 100 warm-up iterations, and 500 timed iterations on one NVIDIA GeForce RTX 3070 with the locked PyTorch 2.11.0/CUDA 12.8/cuDNN 9.19 environment. MACs and parameters are architecture-level monotonic invariants; wall-clock latency is backend and hardware dependent and must be remeasured on each deployment target with `tools/benchmark.py`. The committed regression tests enforce the exact capacity structure rather than a noisy timing threshold.

The B0 bootstrap file `ckpts/PPHGNetV2_B0_stage1.pth` remains valid for all four variants because A/F/P load only shape-matched HGNet core tensors and initialize the synthetic P5 locally, while N still uses the complete B0 core. Previous A/F/P/N full-model checkpoints must not be resumed into this revised scale: A/F/P changed P5 and/or detector dimensions, and N changed query count. Exact-resume validation rejects the old P5 schema and any `num_queries`, `dec_layers`, or `eval_idx` mismatch.

T uses HGNetV2-B1's native stage 4 with the original [LINEA-S detector dimensions](https://github.com/SebastianJanampa/LINEA/blob/master/configs/linea/linea_hgnetv2_s.py): 640 input, hidden width 256, FFN width 512, eight heads, three decoder layers, 1,100 queries, and top-300 output. Its exact fused deploy count is 8,433,105, preserving `N < T < S` while remaining below 8.6M. Download the immutable [B1 stage-1 bootstrap](https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth) explicitly as `ckpts/PPHGNetV2_B1_stage1.pth`; preflight verifies its manifest hash and never downloads it. T has not been fully trained or evaluated in this repository, so no sAP or latency result is claimed.

## Default epoch budgets

The defaults are capacity-aware starting estimates, not empirically established optima. Wireframe train contains 5,000 images, so the committed single-GPU batch-8 profile performs 625 optimizer steps per epoch. Every full recipe applies cosine LR decay to `1e-7` over its complete optimizer-step horizon and selects `checkpoint_best.pth` by validation sAP10.

The estimates combine three reference anchors. The original N recipe supplies N's 72 epochs. There are no A/F/P measurements, so their no-KD values remain conservative 72/66/60 starting points. T uses 60 epochs as the midpoint between N and S; this remains an unvalidated estimate. The DINO values follow the 45/45/40/35 trend for S/M/L/X. XL uses 36 epochs as its sufficient default. The accuracy-tier 2XL and 3XL defaults are both 50 epochs, which is considered sufficient for their non-distillation training; at effective batch 8 this gives each model 31,250 optimizer updates. Their accuracy-first standard configs train the entire DINO core from epoch 0 instead of spending part of that budget on progressive unfreezing. Direct-XL KD now inherits the corresponding no-KD epoch count so its optimizer-step and cosine-LR horizon are directly comparable; the former shorter 60/55/50/50/45/40/40/30/30 ladder was removed after N/S runs showed an early KD regression followed by premature LR decay.

| Variant | No-distillation config | Epochs | Steps | Direct-XL distillation config | Epochs | Steps |
| --- | --- | --: | --: | --- | --: | --: |
| A | `configs/lineae/lineae_a.py` | 72 | 45,000 | `configs/lineae/distill/lineae_a.py` | 72 | 45,000 |
| F | `configs/lineae/lineae_f.py` | 66 | 41,250 | `configs/lineae/distill/lineae_f.py` | 66 | 41,250 |
| P | `configs/lineae/lineae_p.py` | 60 | 37,500 | `configs/lineae/distill/lineae_p.py` | 60 | 37,500 |
| N | `configs/lineae/lineae_n.py` | 72 | 45,000 | `configs/lineae/distill/lineae_n.py` | 72 | 45,000 |
| T | `configs/lineae/lineae_t.py` | 60 | 37,500 | `configs/lineae/distill/lineae_t.py` | 60 | 37,500 |
| S | `configs/lineae/lineae_s.py` | 45 | 28,125 | `configs/lineae/distill/lineae_s.py` | 45 | 28,125 |
| M | `configs/lineae/lineae_m.py` | 45 | 28,125 | `configs/lineae/distill/lineae_m.py` | 45 | 28,125 |
| L | `configs/lineae/lineae_l.py` | 40 | 25,000 | `configs/lineae/distill/lineae_l.py` | 40 | 25,000 |
| X | `configs/lineae/lineae_x.py` | 35 | 21,875 | `configs/lineae/distill/lineae_x.py` | 35 | 21,875 |
| XL | `configs/lineae/lineae_xl.py` | 36 | 22,500 | not applicable (supervised teacher) | — | — |
| 2XL | `configs/lineae/lineae_2xl.py` | 50 | 31,250 | not applicable (supervised teacher) | — | — |
| 3XL | `configs/lineae/lineae_3xl.py` | 50 | 31,250 | not applicable (supervised teacher) | — | — |

Steps assume the default one-GPU batch-8 profile; DDP changes steps per rank but not images seen per epoch. Every KD/no-KD pair now has equal compute and the same cosine-LR horizon by default. S follows the same top-level convention as every other formal variant: `configs/lineae/lineae_s.py` is its full no-KD recipe. The bounded batch-1 diagnostic remains available separately at `configs/lineae/probes/lineae_s.py`.

## LAB and backbone unfreezing

LAB (`LearnableAffineBlock`) applies a learned scalar scale and bias after an activated HGNetV2 `ConvBNAct`. It is enabled for every HGNet student and is not part of DINOv3. The synthetic A/F/P P5 uses average pooling and pointwise Conv-BN-ReLU without LAB.

| Variant | LAB | LAB modules | LAB scalar parameters | Backbone-core schedule |
| --- | --- | --: | --: | --- |
| A | enabled | 20 | 40 | all HGNet stages trainable from epoch 0 |
| F | enabled | 20 | 40 | all HGNet stages trainable from epoch 0 |
| P | enabled | 25 | 50 | all HGNet stages trainable from epoch 0 |
| N | enabled | 30 | 60 | all HGNet stages trainable from epoch 0 |
| T | enabled | 30 | 60 | all HGNet stages trainable from epoch 0 |
| S / M / L / X / XL | not applicable | 0 | 0 | progressive 12-block DINO schedule |
| 2XL | not applicable | 0 | 0 | all 24 DINO blocks trainable from epoch 0 |
| 3XL | not applicable | 0 | 0 | all 32 DINO blocks trainable from epoch 0 |

Progressive unfreezing is implemented for every DINO recipe and follows the configured GazeLLE-style hold semantics. The last two transformer blocks are trainable during epochs 0--4; epoch 5 adds the next earlier block, one earlier block is then added every two epochs, and all 12 blocks are trainable from epoch 23 through the recipe's final epoch. `initial_freeze_epochs=5` therefore means “hold the initial trainable suffix for five epochs,” not “freeze the entire backbone for five epochs.” The SFP, hybrid encoder, and decoder remain trainable throughout. Optimizer groups include the initially frozen DINO parameters from construction time, so late-unfrozen parameters retain stable optimizer/DDP topology.

Epoch numbers below are zero-based, matching logs and checkpoints. Block indices are also zero-based: partial fine-tuning always enables a suffix from the output side of the 12-block DINO core.

| Epoch range | Trainable blocks | Trainable depth | Newly enabled at range start |
| --- | --- | --: | --- |
| 0–4 | 10–11 | 2/12 | blocks 10 and 11 |
| 5–6 | 9–11 | 3/12 | block 9 |
| 7–8 | 8–11 | 4/12 | block 8 |
| 9–10 | 7–11 | 5/12 | block 7 |
| 11–12 | 6–11 | 6/12 | block 6 |
| 13–14 | 5–11 | 7/12 | block 5 |
| 15–16 | 4–11 | 8/12 | block 4 |
| 17–18 | 3–11 | 9/12 | block 3 |
| 19–20 | 2–11 | 10/12 | block 2 |
| 21–22 | 1–11 | 11/12 | block 1 |
| 23–final | 0–11 | 12/12 | block 0 and every remaining DINO-core parameter |

For S/M partial stages, the class token is trainable with the selected blocks. For L/X/XL, the class token, storage tokens, and final normalization are also trainable. Patch embedding and all other core parameters remain frozen until epoch 23. SFP, encoder, and decoder parameters are outside this depth count and train from epoch 0.

The 2XL/3XL accuracy-tier configs deliberately do not use that schedule. They set `progressive_unfreeze=False`, `backbone_trainable_layers=0`, `initial_freeze_epochs=0`, and `unfreeze_interval=0`, so patch embedding, tokens, every DINO block, final normalization, SFP, encoder, and decoder are trainable throughout all 50 epochs. This is part of their standard recipe, not a diagnostic override.

The final fully unfrozen span differs because each recipe has a different epoch budget:

| Variant/recipe | Config | Total epochs | Fully unfrozen epoch range | Fully unfrozen epochs |
| --- | --- | --: | --- | --: |
| S P0 probe | `configs/lineae/probes/lineae_s.py` | 36 | 23–35 | 13 |
| S no-KD | `configs/lineae/lineae_s.py` | 45 | 23–44 | 22 |
| S direct-XL KD | `configs/lineae/distill/lineae_s.py` | 45 | 23–44 | 22 |
| M no-KD | `configs/lineae/lineae_m.py` | 45 | 23–44 | 22 |
| M direct-XL KD | `configs/lineae/distill/lineae_m.py` | 45 | 23–44 | 22 |
| L no-KD | `configs/lineae/lineae_l.py` | 40 | 23–39 | 17 |
| L direct-XL KD | `configs/lineae/distill/lineae_l.py` | 40 | 23–39 | 17 |
| X no-KD | `configs/lineae/lineae_x.py` | 35 | 23–34 | 12 |
| X direct-XL KD | `configs/lineae/distill/lineae_x.py` | 35 | 23–34 | 12 |
| XL no-KD teacher | `configs/lineae/lineae_xl.py` | 36 | 23–35 | 13 |
| 2XL no-KD teacher candidate | `configs/lineae/lineae_2xl.py` | 50 | 0–49 | 50 |
| 3XL no-KD teacher candidate | `configs/lineae/lineae_3xl.py` | 50 | 0–49 | 50 |

X-teacher cascade and tuning configs inherit the corresponding direct-XL KD schedule. The XL EMA and photometric ablations inherit the normal XL schedule. `configs/lineae/ablations/lineae_xl_frozen.py` is the explicit exception: it disables progressive unfreezing and keeps the entire DINO core frozen for all 36 epochs.

## Setup and preflight

Python 3.11 or newer is required; this lower bound is imposed by the pinned `onnxruntime-gpu==1.26.0` export runtime.

```bash
uv sync --locked --extra dev --extra export
uv run --locked python tools/checkpoint_preflight.py
uv run --locked pytest -q
```

Every direct runtime, development, export, and TensorRT dependency is an exact `==` pin in `pyproject.toml`; the same direct versions are used on every supported Python minor. Image processing uses the GUI-enabled `opencv-python==4.13.0.92`, and torchvision is absent. TensorBoard logging continues to use `tensorboardX==2.6.5`, while the compatible `tensorboard==2.21.0` viewer is installed alongside it. TensorBoard requires Pillow as a private transitive UI/image dependency on supported Python versions; no LINEAE training, evaluation, augmentation, rendering, or deployment source imports PIL, so it cannot alter the OpenCV preprocessing contract. `uv.lock` fixes every transitive artifact and its hash; use `--locked` so setup fails instead of silently re-resolving it. Install the separately pinned TensorRT stack on the deployment host with `uv sync --locked --extra tensorrt`.

The preflight validates all nine bootstrap files in `ckpts/` by SHA-256, tensor count, width, and depth. It uses mmap-backed FakeTensors for structural inspection, so the 1.2 GB ViT-L/16 and 3.2 GB ViT-H+/16 checkpoints are not materialized in host memory, and it never downloads missing weights.

## TensorBoard logging

Training creates TensorBoardX event files directly under the configured `output_dir`; eval-only execution does not create a writer, and only distributed rank 0 writes events. A resumed run opens the same directory with `purge_step=checkpoint.global_step`, preventing stale events at and after the resume boundary from appearing beside the restored history.

Training scalars use the cumulative successful optimizer-step count as their x-axis and are written every 10 successful updates. An AMP-overflow step that skips `optimizer.step()` is not written. With gradient accumulation, the logged losses are from the final microbatch that triggers the update, not an average over the accumulation window. `Loss/*` values are averaged across DDP ranks and already include their configured loss coefficients; they are contributions to the optimized objective rather than raw unweighted losses.

| Tag | Availability | Meaning |
| --- | --- | --- |
| `Loss/total` | Every training recipe | Sum of every reduced, weighted supervised and KD loss component for the logged microbatch. This is the scalar checked for finiteness before backward. |
| `Loss/loss_logits` | Every training recipe | Final decoder output's sigmoid focal classification loss, multiplied by `weight_dict['loss_logits']`. |
| `Loss/loss_line` | Every training recipe | Final decoder output's endpoint-swap-invariant L1 line loss, multiplied by `weight_dict['loss_line']`. Lower is better for all `Loss/*` tags. |
| `Loss/loss_logits_<i>`, `Loss/loss_line_<i>` | Auxiliary decoder loss enabled by default | Weighted supervised loss from intermediate decoder layer `<i>`; the final decoder layer uses the unsuffixed tags. |
| `Loss/loss_logits_interm`, `Loss/loss_line_interm` | Every current training recipe | Weighted auxiliary supervision of the encoder's selected top-K proposals before decoder refinement. |
| `Loss/loss_logits_dn_<i>`, `Loss/loss_line_dn_<i>` | Denoising enabled and the batch contains targets | Weighted loss for decoder layer `<i>` on synthetic noisy line/label queries. |
| `Loss/loss_kd_logits` | Distillation only | Confidence-weighted Bernoulli-logit KL on GT-anchored student/teacher pairs, normalized by the batch's GT count and including `distill_class_weight` plus the current KD warm-up weight. |
| `Loss/loss_kd_line` | Distillation only | Confidence-weighted endpoint-invariant Smooth-L1 on the same GT-anchored pairs, normalized by GT count and including `distill_line_weight` plus the current KD warm-up weight. |
| `Loss/loss_kd_feature` | Only when `distill_feature_weight > 0` | Mean aligned P3/P4/P5 feature loss, including feature and KD warm-up weights. It is absent from all default recipes. |
| `Lr/pg_<i>` | Every training recipe | Current AdamW learning rate for optimizer parameter group `<i>`, after warm-up/cosine scheduling. |
| `GradNorm/pg_<i>` | Every training recipe | L2 norm of the parameter group's accumulated gradients after AMP unscaling and before global gradient clipping. A frozen DINO group can report zero until its scheduled unfreeze. |

The current optimizer group indices are stable across variants:

| Group | Parameters |
| --: | --- |
| `pg_0` | Backbone-core weights excluding normalization parameters and biases; uses the configured backbone learning rate. |
| `pg_1` | Backbone-core normalization parameters and biases; uses the backbone learning rate with zero weight decay. |
| `pg_2` | Encoder/decoder normalization parameters and biases; uses the main learning rate with zero weight decay. |
| `pg_3` | All remaining eligible parameters, including the SFP or synthetic P5, ordinary encoder/decoder weights, heads, embeddings, and optional feature-KD projections; uses the main learning rate and weight decay. |

The loss logger is dynamic: if a non-default criterion emits `loss_lmap`, `loss_fgl`, `loss_ddf`, `_pre`, or `_pre_dn` keys, they appear automatically as `Loss/<key>`. Current committed defaults use only classification, line, auxiliary, denoising, and optional KD keys listed above.

| Distillation tag | Meaning |
| --- | --- |
| `Distillation/weight` | Effective KD multiplier at this step: `distill_weight` times the linear `distill_warmup_steps` ramp. |
| `Distillation/temperature` | Teacher/student logit temperature. Improved defaults keep it fixed at 1; explicit legacy experiments may still schedule it. |
| `Distillation/matches` | Teacher proposals accepted by the GT distance gate and attached to the supervised student query for that same GT. |
| `Distillation/candidates` | Teacher proposals remaining after class-0 confidence filtering and `distill_top_k`, before teacher-to-GT matching. |
| `Distillation/rejected` | Candidate proposals not accepted as GT-anchored KD pairs, including unmatched, duplicate, and distance-gate failures. |
| `Distillation/target_coverage` | Accepted KD pairs divided by GT lines in the current microbatch. |
| `Distillation/mean_confidence` | Mean class-0 teacher confidence of accepted pairs. |
| `Distillation/match_weight_sum` | Sum of accepted confidence weights after applying `distill_confidence_power`; this exposes the effective KD mass before GT-count normalization. |
| `Distillation/overhead_ms` | Rank-0 host elapsed time from starting teacher inference through return of the KD criterion, including matching and its GPU-to-CPU synchronization; it is a per-microbatch diagnostic, not a synchronized end-to-end GPU benchmark. |

Validation scalars use the zero-based epoch number as their x-axis and are normally written after each completed epoch when evaluation is enabled. A bounded mid-epoch run can also write them at the current epoch index unless `--skip_eval` is set. Validation losses are dataset averages after DDP synchronization; sAP values are percentages, so higher is better. Each validation pass reports two explicitly labelled protocols: official-compatible sAP over all `num_queries` predictions and deployment sAP over the class-0 top `num_select` predictions. The console prints both groups, JSONL and TensorBoard use explicit `official_sap*` and `deploy_sap*` names, and the canonical `sap*` aliases mean official all-query sAP. `checkpoint_best.pth` is selected by canonical official `sap10` unless `selection_metric` is explicitly changed.

| Validation tag | Meaning |
| --- | --- |
| `Test/loss` | Sum of the weighted validation loss components. |
| `Test/loss_logits` | Weighted final-output focal classification loss on validation data. |
| `Test/loss_line` | Weighted final-output endpoint-invariant L1 line loss on validation data. |
| `Test/sap5` | Official-compatible all-query structural average precision at line-distance threshold 5 in the evaluator's 128-coordinate system. |
| `Test/sap10` | Official-compatible all-query structural average precision at threshold 10; this is the default checkpoint-selection metric. |
| `Test/sap15` | Official-compatible all-query structural average precision at threshold 15. |
| `Test/official_sap5`, `Test/official_sap10`, `Test/official_sap15` | Explicit aliases of the three canonical official-compatible all-query metrics; these remain unambiguous when inspecting a run resumed from a legacy checkpoint. |
| `Test/deploy_sap5` | Deployment structural average precision at threshold 5 after retaining only the class-0 top `num_select` predictions. |
| `Test/deploy_sap10` | Deployment structural average precision at threshold 10 after retaining only the class-0 top `num_select` predictions. |
| `Test/deploy_sap15` | Deployment structural average precision at threshold 15 after retaining only the class-0 top `num_select` predictions. |

### Best-epoch validation renders

After a completed training epoch improves `selection_metric`, rank 0 uses the same normally selected or EMA evaluation model to render the first 10 validation samples. Each PNG shows every ground-truth line in green on the left and class-0 predictions in red on the right. Predictions must have score at least `0.3` and are limited to the best 100 lines. The fixed sample order makes changes directly comparable between best epochs. Ordinary non-best epochs, partial epochs, `--skip_eval`, and standalone `--eval` do not render images.

Images are written atomically to `outputs/<run>/validation_renders/best_epoch_XXXX/NN_image_<image_id>.png`. Only directories matching `best_epoch_XXXX` participate in retention, and the newest 10 best-update epochs are kept across both uninterrupted and resumed training. A rendering failure occurs after `checkpoint_best.pth` is safely written, does not stop training, and is reported through the console plus `validation_render_error` in `log.txt`; successful epochs record `validation_render_dir`.

The comparison canvas, line overlays, labels, and PNG encoding are produced by OpenCV; validation rendering does not introduce a second image backend.

The shared defaults are `validation_render_count=10`, `validation_render_keep_best=10`, `validation_render_score_threshold=0.3`, and `validation_render_max_predictions=100`. Override them through the normal `--options` mechanism; for example, `--options validation_render_count=0` disables rendering without changing validation or checkpoint selection.

Console/JSONL-only diagnostics are not silently represented in TensorBoard: epoch-averaged training meters, AMP overflow counts, KD cache hits/misses/writes, peak CUDA memory, epoch duration, parameter count, best epoch/metric, validation-render status, and detailed profiling timings remain in stdout, `log.txt`, manifests, or dedicated profiling reports.

## S one-batch probe

```bash
python main.py -c configs/lineae/probes/lineae_s.py \
--coco_path data/wireframe_processed --amp --num_workers 0 \
--max_train_steps 1 --skip_eval --skip_profile --verify_optimizer_step \
--options output_dir=outputs/lineae_s_smoke
```

Normal runs omit the bounded/skip flags. Every output directory receives a resolved config, run manifest, annotation/checkpoint hashes, exact backbone load report, TensorBoard events, JSONL log, atomic latest full-state `checkpoint.pth`, and validation-sAP10-selected `checkpoint_best.pth`. Numbered periodic snapshots such as `checkpoint0009.pth` and `checkpoint0019.pth` are disabled by the shared `save_checkpoint_interval=0` default, preventing redundant storage use while retaining exact resume and best-model selection. Set a positive interval explicitly through `--options` only when archival snapshots are required; existing numbered files are not deleted automatically. If `--max_train_steps` stops inside an epoch, the saved diagnostic checkpoint is marked `epoch_complete=False`; it cannot be resumed or promoted as a teacher. Resume always starts after a completed epoch, so it never silently skips the unseen remainder of a bounded probe.

### Exact epoch-boundary resume

`--resume` is a full training-state resume when given the latest `checkpoint.pth`; previously created completed-epoch numbered checkpoints remain loadable even though new ones are disabled by default. Resume strict-loads every model parameter/buffer and restores AdamW state, LR scheduler, warm-up scheduler, GradScaler, EMA, best metric/epoch, global optimizer step, progressive-unfreeze position, and Python/NumPy/Torch/CUDA RNG state for each distributed rank. The frozen teacher is not duplicated into every student checkpoint; its qualified artifact path and SHA-256 are bound by the saved config and revalidated. The current checkpoint schema requires every one of those state fields, and a missing or unexpected scheduler/warm-up/scaler/EMA state fails before any model parameter is mutated rather than silently continuing with a fresh component.

Use the identical config, seed, worker/world-size settings, AMP mode, and output directory. For example:

```bash
uv run --locked python main.py \
-c configs/lineae/distill/lineae_x.py \
--coco_path data/wireframe_processed --device cuda --amp \
--num_workers 8 --seed 42 \
--resume outputs/lineae_x_distill-seed42/checkpoint.pth \
--options output_dir=outputs/lineae_x_distill-seed42
```

Partial-epoch checkpoints (`epoch_complete=False`) are deliberately rejected, because sampler and gradient-accumulation position cannot be reconstructed by skipping unseen samples. Also, a final X-distillation checkpoint from completed epoch 29 already has `start_epoch=30`; with its unchanged 30-epoch config there is no additional work. The current exact-resume contract does not reinterpret or extend a completed LR/KD schedule—choose the intended total epoch budget before starting a matched run.

### Fresh training from best full-model weights

`--init-checkpoint` starts a new run from the full-model inference weights selected by a completed LINEAE checkpoint. Unlike `--resume`, it does not restore AdamW moments, the LR/warm-up scheduler, GradScaler, source epoch/global step, best metric, EMA update count, or RNG state. The new config, seed, optimizer, schedule, unfreezing policy, and distillation setting are authoritative, and training starts at epoch 0/global step 0. `--init-checkpoint` is mutually exclusive with `--resume` and standalone `--eval`; evaluation continues to use `--eval --resume <checkpoint>`.

The loader requires checkpoint format 2, a completed epoch, the current OpenCV preprocessing schema, and the same variant and model semantics. Every shared state key must match exactly in name, shape, and dtype. It uses the checkpoint's selected inference state, so an EMA-selected best checkpoint initializes from EMA weights. The source backbone bootstrap is not loaded redundantly. Initialization provenance, SHA-256, loaded tensor count, newly initialized keys, and ignored source-only keys are written to the console, `resolved_config.json`, `run_manifest.json`, and every new training checkpoint, while optimizer and scheduler state always begin fresh.

Feature KD has one narrow, explicit exception to the full-state key match. A no-KD/output-KD checkpoint has no `distill_feature_projections.*` tensors, while a new run with `distill_feature_weight > 0` adds three train-only 1x1 projection layers. In that case all shared backbone/encoder/decoder tensors are still strict-loaded and only the six projection weight/bias tensors remain at their deterministic fresh initialization. The inverse transition may ignore only source-side `distill_feature_projections.*` tensors when the new run disables feature KD. Any other missing or unexpected key, and every shared shape/dtype mismatch, remains a hard error; this is not a general `strict=False` escape hatch. The feature projections are trained by the new optimizer and KD warm-up and are removed from the deployment graph.

The available completed no-distillation N and S best checkpoints can initialize separate fresh no-KD runs as follows. Set `VARIANT` to exactly `n` or `s`:

```bash
VARIANT=n
uv run --locked python main.py \
-c "configs/lineae/lineae_${VARIANT}.py" \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 43 \
--init-checkpoint \
"outputs/lineae_${VARIANT}-nokd-full-unfreeze-seed42/checkpoint_best.pth" \
--options \
output_dir="outputs/lineae_${VARIANT}-nokd-init-best-seed43"
```

This command deliberately takes the new config's normal unfreezing policy. Add the documented full-unfreeze overrides when the new run should instead keep every backbone block trainable from epoch 0. Changing to another variant is rejected; cross-variant transfer remains a distillation task rather than a partial checkpoint load.

### Fresh accuracy head from a trained DINO core

`--init-backbone-checkpoint` is the narrow migration path from an older same-variant LINEAE checkpoint to a model whose SFP or detector head has changed. It requires checkpoint format 2, a completed epoch, the current OpenCV preprocessing schema, the same variant, and the same DINO core architecture. The loader takes the checkpoint's selected inference state (`model` or `ema_model`) and requires every `backbone.core.*` key, shape, and dtype to match exactly. It intentionally ignores the old SFP, hybrid encoder, decoder, and training-only tensors; those modules receive the new config's normal fresh initialization.

This is a new epoch-0 run. Optimizer moments, schedulers, GradScaler, epoch/global step, best score, EMA counters, and RNG state are not restored. `--init-backbone-checkpoint` is mutually exclusive with `--resume`, `--init-checkpoint`, and standalone `--eval`. The source SHA-256, completed epoch, selected inference state, loaded tensor count, and ignored tensor count are recorded in `resolved_config.json`, `run_manifest.json`, and every checkpoint made by the new run. Subsequent `--resume` restores the new checkpoint completely and does not reopen the migration source.

For example, this transfers only the trained ViT-L/16 core from an old 2XL best checkpoint into the new wide-head 2XL model:

```bash
MODEL=2xl
uv run --locked python main.py \
-c "configs/lineae/lineae_${MODEL}.py" \
--init-backbone-checkpoint \
"outputs/lineae_${MODEL}-nokd-full-unfreeze-seed42/checkpoint_best.pth" \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--options output_dir="outputs/lineae_${MODEL}-wide-core-init-seed42"
```

A–XL full-training configs target one 80--96 GiB GPU with physical batch 8 and no accumulation; 2XL/3XL instead use smaller physical batches plus accumulation to preserve the same effective batch 8. All use LINEA-style batch multi-scale training, while the dedicated S probe above remains fixed at 640 and batch 1 by design. CUDA training DataLoaders pin image/target tensors and use configurable worker prefetching for non-blocking host-to-device transfer. Worker tensor sharing defaults to PyTorch's `file_system` strategy because a large detection batch contains many variable-length target tensors and the `file_descriptor` strategy can exceed the process open-file limit; `multiprocessing_sharing_strategy` is recorded in the resolved config, run manifest, and resume contract. Multi-scale choices are filtered so P3/P4/P5 always contain at least the variant's configured `num_queries` encoder tokens; the actual weighted scale list is stored in checkpoints and run provenance. Training and KD consume every `num_queries` prediction. PyTorch validation computes official-compatible all-query sAP and deployment top-`num_select` sAP together, while postprocessing, ONNX/TensorRT output, end-to-end Torch timing, and ONNX sAP parity apply the class-0 top `num_select` contract. This keeps published-accuracy comparison separate from the configured 200--500-output deployment tuning axis. CUDA training uses fused AdamW, while CPU diagnostics automatically retain the same AdamW equations without requesting the CUDA kernel. DINO activation checkpointing skips redundant RNG snapshots because every checkpointed block is deterministic; RoPE coordinate randomness is generated outside those blocks. A successful full run additionally writes a hash-bound `run_complete.json`, allowing orchestration to distinguish a finished run from an interrupted checkpoint.

All compact and official DINO variants cache the most recent deterministic RoPE sin/cos tensors during evaluation. The cache is keyed by resolution, device, dtype, normalization mode, and the periods-buffer version, and is bypassed for training, tracing, and ONNX export. This accelerates both fixed-shape deployment and repeated online-teacher inference without changing stochastic training or exported graphs. RoPE sin/cos remain at their natural half-head width and are applied directly to the two value halves, rather than duplicating them before every block. This halves their generated and cached tensor footprint while remaining bit-identical to the legacy full-width equation on CPU and CUDA. Full-width inputs remain supported. During Torch `no_grad` inference, compact and official DINO attention also writes those same rotated patch values back into the fresh Q/K projection storage. This removes the per-block rotated-patch and prefix-concatenation tensors; training, tracing, and ONNX export retain the ordinary functional expression.

The shared LINEA decoder also caches its device-local, fixed sinusoidal frequency vector instead of rebuilding it in every decoder layer. Batch expansion for anchors, learned queries, and top-K gather indices uses zero-stride views instead of materialized `repeat` copies. These changes preserve checkpoint keys and apply to training, online KD, Torch inference, and ONNX deployment for every variant. During multi-scale training, deterministic decoder anchors are generated through the legacy CPU math once per feature shape/device and retained in a 16-entry LRU. The 640-base full recipe has 11 unique scales, so normal runs avoid repeated grid construction and host-to-device copies without unbounded cache growth. The hybrid encoder follows the same policy for its deterministic 2D sinusoidal position embeddings. Fixed eval embeddings are non-persistent buffers that move with the model once; dynamic training shapes use a 16-entry device-aware LRU. Neither path adds checkpoint keys. The hybrid encoder also requests no unused attention weights from `nn.MultiheadAttention`, enabling PyTorch's SDPA path for its P5 self-attention. This applies to every backbone family and preserves the existing parameters, masks, residuals, and normalization order. Decoder self-attention uses the same no-weights SDPA path, which matters for the variant-scaled detection queries plus denoising queries processed at each decoder layer; boolean denoising masks retain their semantics. Line deformable attention keeps its original point and reduction order. In `no_grad` forwards only, including online frozen-teacher inference and deployment, the freshly concatenated sampled-value tensor is weighted in place. Training retains the original out-of-place autograd expression, while inference avoids an equally sized temporary allocation.

## Default augmentation

Training applies the same geometry-aware pipeline to no-KD and KD runs, and the teacher receives the exact already-augmented student image. Line endpoints and line maps are transformed together with the pixels.

| Stage | Default behavior |
| --- | --- |
| Flip | Randomly choose the horizontal or vertical flip operator; the selected operator applies with probability 0.5 (25% horizontal, 25% vertical, 50% unchanged overall). |
| Resize/crop | With probability 0.5, resize directly to the variant input. Otherwise resize to 400, 500, or 600, take a random 384--600 crop with line clipping/filtering, then resize to the variant input. Every image resize uses standard OpenCV `INTER_LINEAR`. |
| Color | The OpenCV implementation of LINEA `ColorJitter` always varies brightness, contrast, saturation, and hue in random order (`0.4` magnitude each). |
| Tensor/normalize | Convert to tensor, then use LINEA mean/std for A/F/P/N/T and ImageNet mean/std for S/M/L/X/XL/2XL/3XL. |
| Batch multi-scale | Full recipes resize each assembled batch to a token-safe random size around 75--125% of the base input, with the base size weighted four times. The S P0 probe alone disables this; full S no-KD/KD enables it. |
| Denoising queries | Training adds 300 model-side denoising queries with line noise scale `1.0` and label-noise ratio `0.5`; validation/inference does not. Endpoint noise is additive around each target segment, and a zero line-noise scale is guaranteed to reproduce the same undirected target segment. |
| Validation/test | Deterministic resize to the configured input followed by the same variant normalization; no random augmentation. |

`use_photometric_distort=True` is an explicit ablation, not a default. It replaces the LINEA `ColorJitter` with an OpenCV implementation of the Gazelle-derived image-only distortion. Brightness, contrast, saturation, hue, contrast ordering, and channel permutation retain their existing probabilities while leaving line geometry unchanged.

### OpenCV image preprocessing contract

The fixed preprocessing schema is `opencv_rgb_inter_linear_v2`. Images are decoded by OpenCV with EXIF orientation ignored so COCO coordinates continue to address the stored pixel matrix, converted exactly once from BGR/HWC/uint8 to RGB/HWC/uint8, resized with standard `cv2.INTER_LINEAR`, converted to RGB/CHW/float32 in `[0, 1]`, and normalized with the variant's configured mean/std. `INTER_NEAREST`, `INTER_NEAREST_EXACT`, and `INTER_AREA` are not used for input images. The [UHD downsampling comparison](https://github.com/PINTO0309/UHD#the-impact-of-image-downsampling-methods) ranks LINEAR above NEAREST for accuracy while retaining substantially lower resize cost than AREA. LINEAE therefore uses LINEAR for the external image path and requires deployment preprocessing to reproduce OpenCV's rule exactly.

Batch multi-scale and the online distillation teacher operate on tensors that are already normalized. They use PyTorch `mode="bilinear", align_corners=False`, which follows the same half-pixel sampling rule as OpenCV `INTER_LINEAR` and is tested against it within float32 tolerance, avoiding a GPU-to-CPU round trip. This training-only tensor interpolation is not exported. Line-map supervision and internal feature-map alignment retain their explicit PyTorch interpolation rules, while inference-time FPN feature upsampling deliberately remains nearest-neighbor for a simple quantization-friendly graph.

The exported ONNX graph continues to accept normalized RGB/NCHW/float32 and does not embed decoding or resizing. Python deployment can share the exact training path as follows; other runtimes must reproduce the same ordered operations.

```python
from util.image_preprocess import preprocess_image_file

images = preprocess_image_file(
    "input.jpg",
    size_hw=(640, 640),
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
).unsqueeze(0).numpy()
```

Detector checkpoints use checkpoint format 2 and record the preprocessing schema. Evaluation, export, benchmark, qualification, and resume reject artifacts that lack or disagree with the current OpenCV schema, so Pillow-trained and `opencv_rgb_inter_nearest_v1` detector checkpoints, existing qualified teachers, evaluation reports, ONNX models, and teacher caches must be regenerated. External `INTER_LINEAR` preprocessing is outside the ONNX graph and does not prevent Conv/Linear layers from being quantized. This does not affect the HGNetV2 or DINOv3 backbone initialization weights in `ckpts/`.

### Why the dataset remains files rather than one Parquet file

One monolithic Parquet file is not recommended for this dataset. The copied tree is only 5,568 files / about 1.89 GB, JPEG decoding and geometry augmentation still dominate CPU work, and a single container would make random multi-worker/DDP reads contend on one file without removing image decode cost. It also makes incremental replacement and recovery coarser and would add PyArrow to the fixed runtime.

Parquet can help when metadata filtering or remote columnar analytics dominates, but that is not this training access pattern. If profiling on network/object storage later shows filesystem metadata as the bottleneck, prefer multiple deterministic shards (Parquet row groups or tar/WebDataset shards), preserve the COCO image/annotation identity in the run manifest, and benchmark throughput and shuffle quality against the current loader before changing the default.

### Annotation visualization

`data/annotation_rendering.py` renders every GT line from a COCO line-annotation JSON directly over the original, unresized image with OpenCV. It interprets each stored `line` as `[x, y, dx, dy]`, converts it to `(x, y) -> (x + dx, y + dy)`, and clips both endpoints exactly as the training dataset does. With no arguments it renders all 462 Wireframe validation images as PNG files under `data/test_render/`:

```bash
uv run --locked python data/annotation_rendering.py
```

The output directory is protected against accidental replacement. Use `--overwrite` to regenerate it, or `--limit` to render only the first records for a quick inspection:

```bash
uv run --locked python data/annotation_rendering.py --limit 10 --overwrite
```

York validation or Wireframe training annotations use the same renderer by selecting their JSON and image directory. Choose a distinct output directory when retaining multiple rendered sets:

```bash
uv run --locked python data/annotation_rendering.py \
--annotation-file data/york_processed/annotations/lines_val2017.json \
--image-dir data/york_processed/val2017 \
--output-dir /tmp/lineae_york_annotation_render

uv run --locked python data/annotation_rendering.py \
--annotation-file data/wireframe_processed/annotations/lines_train2017.json \
--image-dir data/wireframe_processed/train2017 \
--output-dir /tmp/lineae_wireframe_train_annotation_render
```

## A–X supervised workflow without distillation

A, F, P, N, T, S, M, L, and X can all be trained without a teacher through the same top-level `configs/lineae/lineae_<variant>.py` convention; do not use anything under `configs/lineae/distill/`. Every listed config has `distill_weight=0.0`, so neither `ckpts/lineae_xl_teacher.pth` nor any other qualified teacher artifact is required.

| Variant | Supervised config | Default epochs |
| --- | --- | --: |
| A | `configs/lineae/lineae_a.py` | 72 |
| F | `configs/lineae/lineae_f.py` | 66 |
| P | `configs/lineae/lineae_p.py` | 60 |
| N | `configs/lineae/lineae_n.py` | 72 |
| T | `configs/lineae/lineae_t.py` | 60 |
| S | `configs/lineae/lineae_s.py` | 45 |
| M | `configs/lineae/lineae_m.py` | 45 |
| L | `configs/lineae/lineae_l.py` | 40 |
| X | `configs/lineae/lineae_x.py` | 35 |

Set `VARIANT` to exactly one of `a`, `f`, `p`, `n`, `t`, `s`, `m`, `l`, or `x` and start that single run. The committed config supplies its capacity-aware epoch budget, batch 8, no accumulation, multi-scale training, cosine LR, backbone initialization, and progressive unfreezing where applicable:

```bash
VARIANT=x
uv run --locked python main.py \
-c "configs/lineae/lineae_${VARIANT}.py" \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--options output_dir="outputs/lineae_${VARIANT}-nokd-seed42"
```

The S backbone initialization file is named `ckpts/vitt_distill.pt`, but it is only a pretrained backbone weight. Its filename does not enable online teacher inference or a distillation loss in this supervised run. Use the separately documented `configs/lineae/probes/lineae_s.py` only for a bounded batch-1 pipeline diagnostic.

## XL teacher workflow

Train XL supervised first. The recommended single-96-GB-GPU recipe is batch 8, no accumulation, 36 epochs, cosine LR, AMP, eight workers, seed 42, and the progressive schedule documented above. The 36-epoch LR horizon is calibrated to 625 optimizer updates per epoch and 22,500 updates in total; increasing `batch_size_train` does not automatically scale either LR, so batch 64 would provide only 78 updates per epoch and 2,808 updates in total and is not the committed teacher recipe. Increase `batch_size_val` independently when it fits. Disabling activation checkpointing trades spare VRAM for training speed without changing optimization semantics.

Warmup remains disabled by default. When an experiment explicitly sets `use_warmup=True`, `warmup_iters` is now contained within the fixed optimizer-step horizon: the downstream cosine duration is shortened by the warmup interval and still reaches `min_lr` at the final checkpoint. For the batch-8 XL horizon, `warmup_iters=3125` means five warmup epochs inside the same 22,500 total updates, not five additional epochs. Invalid warmup durations that consume the entire run fail before training.

```bash
uv run --locked python main.py \
-c configs/lineae/lineae_xl.py \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--options \
output_dir=outputs/lineae_xl-seed42 \
batch_size_train=8 \
batch_size_val=64 \
epochs=36 \
gradient_accumulation_steps=1 \
use_checkpoint=False
```

To resume an interrupted XL run from the latest completed epoch, use the same config, data path, device topology, AMP mode, worker count, seed, output directory, batch settings, and total epoch budget:

```bash
uv run --locked python main.py \
-c configs/lineae/lineae_xl.py \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--resume outputs/lineae_xl-seed42/checkpoint.pth \
--options \
output_dir=outputs/lineae_xl-seed42 \
batch_size_train=8 \
batch_size_val=64 \
epochs=36 \
gradient_accumulation_steps=1 \
use_checkpoint=False
```

`checkpoint.pth` is the atomic latest full-state checkpoint and resumes at the epoch after its saved completed epoch. The model, AdamW, cosine scheduler, GradScaler, progressive-unfreeze position, best metric/epoch, global optimizer step, and all RNG states are restored. A checkpoint saved after epoch 35 has already completed the 36-epoch recipe and therefore has no remaining training work; `--resume` does not extend the schedule.

Do not resume training from a checkpoint created before both the `endpoint_offset_v2` denoising correction and the `undirected_direct_tie_v2` endpoint-loss correction. The first inherited line-noise expression did not preserve the target even when its noise scale was zero. After correcting that issue, the six DN losses learned normally, but ordinary proposals still remained zero-length: epoch 0 and epoch 11 checkpoints produced `1,100/1,100` zero-length lines on every inspected validation image. The cause was `torch.minimum(direct, swapped)` splitting gradients equally when the two endpoint orders tied. LINEA point anchors therefore gave both endpoints identical gradients and could never acquire length. The corrected loss selects the direct branch only on exact ties, retaining the same undirected scalar loss while giving the two endpoint slots opposite gradients. Training resumes without either semantic marker are rejected; eval-only loading remains possible. Restart from the configured DINOv3 initialization weights in a new output directory.

To disable progressive unfreezing and train the entire XL DINO core from epoch 0, start a separate diagnostic run with all four unfreeze controls overridden:

```bash
uv run --locked python main.py \
-c configs/lineae/lineae_xl.py \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--options \
output_dir=outputs/lineae_xl-full-unfreeze-v2-seed42 \
batch_size_train=8 \
batch_size_val=64 \
epochs=36 \
gradient_accumulation_steps=1 \
progressive_unfreeze=False backbone_trainable_layers=0 \
initial_freeze_epochs=0 unfreeze_interval=0 \
use_checkpoint=False
```

In LINEAE, `backbone_trainable_layers=0` means all backbone blocks, while `-1` means none. Setting only `progressive_unfreeze=False` leaves the final two blocks trainable for the entire run instead of enabling all 12. Immediate full unfreezing is not the recommended teacher recipe: it exposes the whole pretrained core to an initially random detection head and can destabilize or erase useful pretrained features. It also allocates backbone gradients and AdamW state from the beginning, so its peak VRAM is higher. This is a separate training recipe and cannot resume a checkpoint created with the progressive settings; keep the distinct `output_dir` shown above. The diagnostic command explicitly disables activation checkpointing to use the available VRAM for speed.

Evaluate both the candidate and the reproduced baseline on both datasets with `tools/evaluate_checkpoint.py`. Then promote only an XL candidate whose recorded Wireframe sAP10 beats the baseline:

```bash
python tools/qualify_teacher.py \
--candidate outputs/lineae_xl-seed42/checkpoint_best.pth \
--candidate-metrics outputs/evaluations/lineae_xl.json \
--baseline-checkpoint outputs/linea_hgnetv2_n/checkpoint_best.pth \
--baseline-metrics outputs/evaluations/linea_hgnetv2_n.json
```

This performs strict reload and identical-output checks before writing `ckpts/lineae_xl_teacher.pth`. Distillation configs fail immediately while that qualified checkpoint is absent.

## 2XL / 3XL accuracy workflow

2XL and 3XL are supervised accuracy-tier teacher candidates and do not replace the qualified XL used by existing distillation configs. Both keep 1,100 queries, 300 DN queries, top-300 deployment selection, Official sAP10 best-checkpoint selection, canonical 640 input, the 480–800 batch multi-scale range, and the existing supervised loss composition. Their standard heads are intentionally much larger than XL:

| Setting | 2XL | 3XL |
| --- | ---: | ---: |
| DINO blocks fused | 5 / 11 / 17 / 23 | 7 / 15 / 23 / 31 |
| SFP / hidden width | 512 | 640 |
| Attention heads | 16 | 20 |
| Transformer FFN width | 2,048 | 2,560 |
| P4/P5 encoder layers | 2 independent layers each | 2 independent layers each |
| Decoder layers / `eval_idx` | 8 / 7 | 10 / 9 |
| Decoder sampling points | `[8, 4, 2]` | `[8, 4, 2]` |
| Regression bins / dropout | 32 / 0.1 | 32 / 0.1 |
| FPN/PAN expansion / depth | 0.5 / 1.0 | 0.5 / 1.0 |

P3 bypasses global Transformer attention; separate two-layer encoders operate only on P4 and P5 before the enlarged FPN/PAN. Intermediate DINO fusion uses the final block as the reference and adds the mean of three gated intermediate-minus-final residuals through a Dirac-initialized 1×1 projection. The `tanh` gates start at zero, so the initialized fused map is bit-exactly the final-block map while the gates can learn useful earlier-block corrections. EMA and feature KD remain disabled by default.

On one 96 GB GPU, 2XL uses physical batch 4 with two-step accumulation and 3XL uses physical batch 2 with four-step accumulation. Both retain effective batch 8, 625 optimizer updates per epoch, and a 50-epoch/31,250-update horizon. The entire DINO core is trainable from epoch 0, activation checkpointing is enabled, the backbone LR remains `1e-5`, the head LR remains `2e-4`, cosine decay is used, and warm-up is disabled. After measuring sufficient VRAM headroom, `--options use_checkpoint=False` may be used to trade memory for speed without changing model semantics.

The new deployed counts at 640×640 are 311.5M backbone + 60.7M after-backbone = 372.2M and 1,173.6 GFLOPs for 2XL, and 853.7M + 106.8M = 960.5M and 3,043.2 GFLOPs for 3XL. These are the exact meta-graph values behind the parameter inventory. Full-model checkpoints from the former 256-channel/six-decoder architecture are incompatible with the new standard config for resume, formal evaluation, and `--init-checkpoint`. Only their same-variant DINO core may be migrated through `--init-backbone-checkpoint` as documented above.

Run the bounded optimizer smoke on the 96 GB host before committing to either full schedule. Set `MODEL` to `2xl` or `3xl`; this loads the official DINO bootstrap checkpoint and completes exactly one optimizer update without evaluation, FLOP profiling, or numbered periodic snapshots. It still writes the atomic latest-state `checkpoint.pth` for diagnosis:

```bash
MODEL=2xl
uv run --locked python main.py \
-c "configs/lineae/lineae_${MODEL}.py" \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--max_train_steps 1 \
--skip_eval \
--skip_profile \
--verify_optimizer_step \
--options output_dir="outputs/smoke_lineae_${MODEL}-seed42"
```

The standard single-GPU commands use the committed batch, accumulation, epoch, LR, activation-checkpointing, and full-unfreeze settings. These start from the official DINO bootstrap weights; add the core-migration option shown above when a compatible old same-variant best checkpoint is available:

```bash
uv run --locked python main.py \
-c configs/lineae/lineae_2xl.py \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--options output_dir=outputs/lineae_2xl-seed42

uv run --locked python main.py \
-c configs/lineae/lineae_3xl.py \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--options output_dir=outputs/lineae_3xl-seed42
```

Resume the atomic full state with the same config and output directory. For 3XL, replace both `2xl` occurrences with `3xl`:

```bash
uv run --locked python main.py \
-c configs/lineae/lineae_2xl.py \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--resume outputs/lineae_2xl-seed42/checkpoint.pth \
--options output_dir=outputs/lineae_2xl-seed42
```

Because the complete core is trainable immediately, the first optimizer step is representative of backbone gradient and AdamW-state allocation; validate that smoke step before a full run. Do not raise the effective batch above eight when tuning physical batch and accumulation. The 3XL state exceeds the practical single-file ONNX protobuf limit; external-data ONNX support is outside this accuracy-training workflow, whose completion artifacts are PyTorch checkpoints and evaluation reports.

## Distillation

After teacher qualification, choose one X-and-smaller variant and run it individually. Set `STUDENT` to exactly one of `x`, `l`, `m`, `s`, `t`, `n`, `p`, `f`, or `a`; this is a one-run command template, not an experiment runner. All nine configs use batch 8, no accumulation, the same epoch and cosine-LR horizon as their no-KD counterpart, GT-anchored output KD, a 6,250-step/10-epoch KD warm-up, and the qualified canonical-640 XL teacher:

```bash
STUDENT=x
uv run --locked python main.py \
-c "configs/lineae/distill/lineae_${STUDENT}.py" \
--coco_path data/wireframe_processed \
--device cuda \
--amp \
--num_workers 8 \
--seed 42 \
--options \
output_dir=outputs/lineae_${STUDENT}_distill-seed42 \
batch_size_train=8 \
batch_size_val=64 \
gradient_accumulation_steps=1 \
distill_weight=1.0 \
distill_teacher_checkpoint=ckpts/lineae_xl_teacher.pth \
distill_matching_mode=gt_anchored \
distill_confidence_threshold=0.5 \
distill_teacher_gt_max_distance=10.0 \
distill_confidence_power=1.0 \
distill_class_weight=0.25 \
distill_line_weight=5.0 \
distill_temperature_start=1.0 \
distill_temperature_end=1.0 \
distill_temperature_steps=0 \
distill_warmup_steps=6250
```

The improved output-KD path first filters and ranks teacher proposals with the same class-0 score as LINEA evaluation. It then performs a one-to-one, endpoint-invariant teacher-to-GT assignment and rejects assigned teacher lines whose evaluator-equivalent squared endpoint distance is not below `distill_teacher_gt_max_distance=10`. The supervised matcher independently determines which student query owns each GT; KD is attached only to that exact query and GT identity. A teacher false positive therefore cannot turn a supervised background query into a KD foreground target, and two assignments cannot direct one query toward different GT lines. Accepted Bernoulli-logit KL and Smooth-L1 line losses are weighted by teacher confidence raised to `distill_confidence_power`, summed, and divided by the batch's distributed GT-count normalizer. This makes lower teacher coverage reduce total KD strength instead of increasing the gradient per surviving match. Classification KD uses weight 0.25 because N/S diagnostics found the legacy class term dominated line KD by roughly 55–57 times. Temperature remains at 1 because `T^2` compensation made the measured legacy `T=1 -> 4` schedule strengthen rather than soften class gradients. The 10-epoch ramp prevents a randomly initialized head and newly trainable backbone blocks from receiving full KD immediately.

HGNet students receive the same augmented pixels with an explicit LINEA-to-ImageNet normalization conversion for the teacher. The frozen teacher evaluates that tensor at its own canonical 640x640 input; normalized line endpoints remain directly comparable. Optional feature KD bilinearly aligns canonical teacher feature maps to each student's current multi-scale feature sizes. All non-empty teacher-to-GT cost matrices stay on the GPU until one flattened CPU transfer, after which SciPy solves each original matrix independently. Supervised and KD endpoint losses select the direct order when directed and swapped losses tie, preventing zero-length point anchors from receiving identical endpoint gradients without changing the undirected scalar. Wireframe and York contain only category 0; the second retained checkpoint channel remains a focal-loss negative and cannot select proposals or influence KD loss. The root-owned LINEA control retains its original directed matcher/loss. Exact-input teacher caching remains transform-safe and unchanged.

For a reproduction of the previous independent proposal-set KD, explicitly set `distill_matching_mode=independent`, `distill_confidence_threshold=0.3`, `distill_confidence_power=0.0`, `distill_class_weight=1.0`, `distill_warmup_steps=1000`, `distill_temperature_end=4.0`, and `distill_temperature_steps=-1`. Checkpoints created with that legacy recipe must not be resumed under the GT-anchored defaults because the optimization contract and LR horizon differ; start a new output directory.

The gated workflow can be rendered as a non-executing plan:

```bash
uv run --locked python tools/plan_experiment_matrix.py --stage distillation \
--output-root outputs/full_matrix_seed42
```

This command never launches training or evaluation. Execute the listed commands individually on the intended training host, following the recorded dependencies; resume a completed-epoch checkpoint explicitly with `main.py --resume` when needed. The plan compares frozen and progressively unfrozen XL recipes and places KD only after the best valid XL candidate has passed qualification. The default all-stage plan is the 115-task core comparison. Add `--include-ablations` to opt into the 131-task plan, which also measures XL EMA and photometric recipes plus X intermediate-fusion and feature-KD recipes through the same accuracy and deployment gates.

Only after the direct-XL controls exist, an intermediate-teacher experiment can be planned with `--stage cascade --include-cascade`. It first requires the XL-distilled X checkpoint to beat the matched X no-KD checkpoint, promotes it as `ckpts/lineae_x_teacher.pth`, and then trains L/M/S/N/P/F/A with that qualified X teacher. T has no cascade config in this change. The cascade-stage plan has 35 tasks; `--stage all` has 158 including all accuracy and deployment measurements. It is never enabled by default.

Coarse per-variant Pareto screening is also opt-in:

```bash
uv run --locked python tools/plan_experiment_matrix.py --stage tuning --include-tuning \
--output-root outputs/full_matrix_seed42
```

For each X/L/M/S/N/P/F/A model this compares the direct-XL recipe with a fixed speed bundle and accuracy bundle. T has no coarse tuning candidates in this change. The candidates vary only input size, query/top-K count, and decoder depth while retaining the same XL teacher and training semantics. The 97-task tuning-stage plan produces hash-matched Wireframe/York metrics, CUDA latency/memory reports, and one Pareto report per tuned variant. The core plan remains unchanged and no tuning candidate is retained without measured repeated-seed evidence.

`tools/analyze_repeated_pareto.py` is an optional read-only post-processor for manually scheduled repeated runs. It does not launch or orchestrate training. It requires at least three matched seeds and reports 95% confidence intervals, paired deltas against the direct-XL baseline, and mean/robust Pareto sets.

Online-teacher training cost can be measured without starting a full run using `tools/profile_training.py`. It performs only the requested warm-up and measured real-data optimizer steps in memory, saves no checkpoint, and records phase timings, throughput, and peak CUDA memory. Run it once for a matched no-KD config and once for its KD config, then pass both JSON files to `tools/compare_training_profiles.py`. The comparison tool is read-only and rejects mismatched variants, data, initialization, hardware, precision, seeds, trainable depth, or measured input-size sequences.

## Measurement and deployment

`tools/evaluate_checkpoint.py` evaluates one checkpoint on any repeatable set of validation roots and can optionally render a fixed prefix from each dataset. The left half of every PNG shows GT in green; the right half shows score-filtered predictions in red. Rendering is disabled by default. `--render-count N` enables it, `--render-score-threshold` controls the minimum class-0 score, and `--render-max-predictions` caps the number of drawn lines after score sorting. Add `--render-endpoints` to mark the start of every GT and prediction line in blue and the end in yellow. Dataset names become output subdirectory names, so the following command writes metrics to one JSON report and ten PNG comparisons each under `..._renders/wireframe/` and `..._renders/york/`:

```bash
uv run --locked python tools/evaluate_checkpoint.py \
-c configs/lineae/lineae_3xl.py \
--checkpoint outputs/lineae_3xl-seed42/checkpoint_best.pth \
--dataset wireframe=data/wireframe_processed \
--dataset york=data/york_processed \
--device cuda \
--amp \
--batch-size 1 \
--num-workers 8 \
--render-count 10 \
--render-score-threshold 0.3 \
--render-max-predictions 100 \
--render-endpoints \
--render-output-dir outputs/evaluations/lineae_3xl-seed42_renders \
--output outputs/evaluations/lineae_3xl-seed42.json
```

If `--render-output-dir` is omitted, the default is `<output-stem>_renders` beside the JSON report. Existing per-dataset render directories are atomically replaced when the command is rerun; unrelated sibling directories remain untouched. Files and symbolic links are never accepted as replacement targets. The JSON report records the effective render settings and each dataset's absolute render directory.

For a visual check without full-dataset sAP evaluation, add `--render-only`. This path does not construct the criterion or evaluation DataLoader, does not hash the checkpoint, and runs inference only on the first `--render-count` samples of each requested dataset. `--output` is optional and, when supplied, receives a diagnostic `lineae_render_v1` manifest rather than a qualification-compatible evaluation report. Render-only permits `--render-max-predictions` up to the model's `num_queries`, independently of the configured deployment `num_select`:

```bash
uv run --locked python tools/evaluate_checkpoint.py \
-c configs/lineae/lineae_3xl.py \
--checkpoint outputs/lineae_3xl-nokd-full-unfreeze-seed42-wide/checkpoint_best.pth \
--dataset wireframe=data/wireframe_processed \
--dataset york=data/york_processed \
--device cuda \
--amp \
--batch-size 1 \
--render-only \
--render-count 100 \
--render-score-threshold 0.2 \
--render-max-predictions 600 \
--render-output-dir outputs/evaluations/lineae_3xl-nokd-full-unfreeze-seed42_renders
```

`tools/benchmark.py` first applies LINEA's fused deploy conversion, then records FLOPs/MACs, parameters, peak memory, raw samples, and batch-1 p50/p95 Torch latency. `tools/export_onnx.py` exports the same fused model, validates it with `onnx.checker`, and simplifies it with onnxsim by default (`--disable-onnxsim` is available for diagnosis); it does not start ONNX Runtime or perform numerical parity. The exported top-k defaults to the config's `num_select`, but `--num-select K` (or its `--topk K` alias) embeds any validated `1 <= K <= num_queries` value into the fixed ONNX output shapes and records the effective and configured values in the `.export.json` report. `tools/benchmark_tensorrt.py` separately builds with TF32 disabled, measures the engine, and gates its actual FP16 outputs against ONNX Runtime using the query/endpoint-order-invariant comparison. `tools/evaluate_onnx.py` separately enforces full-dataset deployment sAP5/10/15 parity against the hash-matched PyTorch evaluation before TensorRT benchmarking; its CUDA ORT mode disables TF32 and CPU execution-provider fallback. For a custom export top-k, create the PyTorch report with the same `tools/evaluate_checkpoint.py --num-select K`; `tools/evaluate_onnx.py` reads K from the hash-bound export report. `tools/analyze_pareto.py` identifies non-dominated variants, and `tools/generate_model_card.py` turns archived reports into model cards. It requires a hash-matched Pareto report and refuses to label a dominated model as a qualified candidate.

For example, this exports XL with 500 retained line queries without changing its training config:

```bash
uv run --locked python tools/export_onnx.py \
-c configs/lineae/lineae_xl.py \
--checkpoint outputs/lineae_xl-seed42/checkpoint_best.pth \
--output outputs/onnx/lineae_xl-top500.onnx \
--num-select 500
```

### Interactive ONNX demo

`demo_lineae.py` runs a LINEAE ONNX model on one image, an image directory, a video, or a camera index. It uses the fixed input size and top-k dimensions exposed by the ONNX model itself and does not require or inspect an export report. The variant is normally inferred from a filename containing `lineae_<variant>`; specify `--variant` for a custom filename. A/F/P/N/T use the LINEA mean/std, while S/M/L/X/XL/2XL/3XL use ImageNet mean/std. All inputs follow the training contract: OpenCV decode with EXIF orientation ignored for still images, BGR-to-RGB conversion, `INTER_LINEAR` resize, RGB/NCHW/float32 conversion, and normalization.

```bash
uv run --locked --extra export python demo_lineae.py \
--input data/wireframe_processed/val2017/00380861.png \
--model outputs/lineae_xl-full-unfreeze-seed42/lineae_xl_1x3x640x640_1100.onnx \
--execution-provider cuda \
--score-threshold 0.4 \
--max-lines 100 \
--disable-display
```

- USBcam
```bash
uv run --locked --extra export python demo_lineae.py \
--input 0 \
--model outputs/lineae_xl-full-unfreeze-seed42/lineae_xl_1x3x640x640_1100.onnx \
--execution-provider tensorrt \
--score-threshold 0.2 \
--max-lines 1100
```

The script applies sigmoid to class-0 logits, converts normalized `[x1,y1,x2,y2]` predictions back to the source image dimensions, and renders the highest-scoring threshold-passing lines. `--max-lines` limits drawing only; the ONNX graph still computes the top-k embedded by `tools/export_onnx.py`. Results default to `output/demo_lineae/`. Use `--execution-provider cpu` or `tensorrt` as needed and `--disable-save` for display-only operation. TensorRT execution always sets `trt_engine_cache_path` to the selected ONNX model's directory, so its generated engine cache is stored beside that model.

## Licensing

LINEAE is distributed under the root [Apache License 2.0](LICENSE).

## Cited / Acknowledgement
- https://github.com/SebastianJanampa/LINEA
  ```bibtex
  @misc{janampa2025linea,
    title={LINEA: Fast and Accurate Line Detection Using Scalable Transformers},
    author={Sebastian Janampa and Marios Pattichis},
    year={2025},
    eprint={2505.16264},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2505.16264},
  }
  ```
- https://github.com/Peterande/D-FINE
  ```bibtex
  @misc{peng2024dfine,
    title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
    author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
    year={2024},
    eprint={2410.13842},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
  ```
- https://github.com/facebookresearch/dinov3
  ```bibtex
  @misc{simeoni2025dinov3,
    title={{DINOv3}},
    author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
    year={2025},
    eprint={2508.10104},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2508.10104},
  }
  ```
- https://github.com/PINTO0309/gazelle-dinov3
  ```bibtex
  @software{Hyodo_2025_gazelle_dinov3,
    author    = {Katsuya Hyodo},
    title     = {gazelle-dinov3: Gaze-LLE-DINOv3},
    year      = {2025},
    month     = {oct},
    publisher = {Zenodo},
    version   = {1.0.0},
    doi       = {10.5281/zenodo.17413165},
    url       = {https://github.com/PINTO0309/gazelle-dinov3},
    abstract  = {A model for activating human gaze regions using heat maps, built with DINOv3.},
  }
  ```
