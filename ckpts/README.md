# Backbone bootstrap checkpoints

This directory contains immutable pretrained **backbone initialization weights** used when a new LINEAE training run starts at epoch 0. They are not trained LINEAE detector checkpoints, resume files, or distillation teachers. LINEAE never downloads them implicitly: obtain the required upstream artifacts, keep the exact local filenames below, and verify them before training.

The authoritative local integrity metadata is [`MANIFEST.json`](MANIFEST.json). The variant-to-file mapping is defined in [`models/lineae/variants.py`](../models/lineae/variants.py).

## Required files and origins

| LINEAE variant | Local filename | Upstream origin | Expected SHA-256 |
| --- | --- | --- | --- |
| A / F / P / N | `PPHGNetV2_B0_stage1.pth` | [D-FINE HGNetV2-B0 stage-1 bootstrap](https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth) | `70a372e8cbc59b34c5da2943261ecb633faf304a58e7e05461a27bd8d8b7f3d1` |
| T | `PPHGNetV2_B1_stage1.pth` | [D-FINE HGNetV2-B1 stage-1 bootstrap](https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth) | `3195a3fbd853af42c0c06eb121338077e96a5f7580daf503a4c05d4fc84b7fb9` |
| S | `vitt_distill.pt` | [DEIMv2 ViT-Tiny distilled from DINOv3-S](https://github.com/Intellindust-AI-Lab/DEIMv2#23-backbone-preparation); [pinned artifact](https://huggingface.co/KeepParallel/deimv2/blob/6f7b53966fca48b6386d62ccaefd4223a022c980/vitt_distill.pt) | `2053b865f4e2673fba3f95f7e7e54ad5ee18143885e3ad27eaabb5b3b9919738` |
| M | `vittplus_distill.pt` | [DEIMv2 ViT-Tiny+ distilled from DINOv3-S](https://github.com/Intellindust-AI-Lab/DEIMv2#23-backbone-preparation); [pinned artifact](https://huggingface.co/KeepParallel/deimv2/blob/6f7b53966fca48b6386d62ccaefd4223a022c980/vittplus_distill.pt) | `470b65d9a7704973ae105057f7ec7a4c85f853bc379a7bfbcc49ecc80a17e25b` |
| L | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | Meta DINOv3 ViT-S/16, LVD-1689M | `08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d` |
| X | `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` | Meta DINOv3 ViT-S+/16, LVD-1689M | `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea` |
| XL | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | Meta DINOv3 ViT-B/16, LVD-1689M | `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c` |
| 2XL | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | Meta DINOv3 ViT-L/16, LVD-1689M | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| 3XL | `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` | Meta DINOv3 ViT-H+/16, LVD-1689M | `7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488` |

A/F/P intentionally load only the shape-compatible HGNetV2-B0 core and initialize their LINEAE-specific synthetic P5 locally. N uses the complete B0 core, and T uses the complete B1 core. The words `distill` in the S/M filenames describe how DEIMv2 produced those pretrained backbones; placing either file here does not enable LINEAE knowledge distillation.

## Download

Run the following commands from the repository root. The revision-pinned DEIMv2 URLs are used so that a later repository update cannot silently change the requested artifact.

```bash
mkdir -p ckpts

wget -O ckpts/PPHGNetV2_B0_stage1.pth \
  https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth
wget -O ckpts/PPHGNetV2_B1_stage1.pth \
  https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth

wget -O ckpts/vitt_distill.pt \
  'https://huggingface.co/KeepParallel/deimv2/resolve/6f7b53966fca48b6386d62ccaefd4223a022c980/vitt_distill.pt?download=true'
wget -O ckpts/vittplus_distill.pt \
  'https://huggingface.co/KeepParallel/deimv2/resolve/6f7b53966fca48b6386d62ccaefd4223a022c980/vittplus_distill.pt?download=true'
```

The five official DINOv3 LVD-1689M checkpoints are access-controlled by Meta. Request access through the [official DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), then use the URLs sent by Meta with `wget` and save each file under the exact local name in the table. The [official DINOv3 repository](https://github.com/facebookresearch/dinov3#pretrained-models) also documents this workflow and recommends `wget` rather than a browser. Do not commit or publish the private, expiring URLs from the access email.

For example:

```bash
wget -O ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  '<VIT-S/16 URL FROM META>'
wget -O ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth \
  '<VIT-S+/16 URL FROM META>'
wget -O ckpts/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  '<VIT-B/16 URL FROM META>'
wget -O ckpts/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  '<VIT-L/16 URL FROM META>'
wget -O ckpts/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
  '<VIT-H+/16 URL FROM META>'
```

Hugging Face Transformers repositories for the official DINOv3 models may use different serialization, key namespaces, or filenames. They are not drop-in replacements for the exact `.pth` files bound by this repository's manifest.

## Verification

After all nine bootstrap files are present, validate their SHA-256, tensor count, model width/depth, and state-dict shapes:

```bash
uv run --locked python tools/checkpoint_preflight.py
```

The preflight is read-only and never downloads missing files. A missing file, hash mismatch, or architecture mismatch is a hard failure. Do not update `MANIFEST.json` merely to accept an unexpected download; first confirm that the upstream artifact and local filename are correct.

The large binary weights are intentionally excluded by the repository's `*.pth` and `*.pt` ignore rules. Only this documentation and `MANIFEST.json` should be committed. Review and comply with each upstream project's model-weight license and usage terms, including the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).

Qualified LINEAE teacher files such as `lineae_xl_teacher.pth` or `lineae_3xl_teacher.pth` are separate, locally produced artifacts governed by the distillation qualification workflow. They are not among the nine downloadable backbone bootstraps above.
