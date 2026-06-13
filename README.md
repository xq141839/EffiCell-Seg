# MICCAI 2026 | Rethinking the Adaptation of Vision Foundation Models for Efficient Cell Segmentation

[[`arXiv`](<ARXIV_URL>)]

-------------------------------------------

<!-- TODO: replace figs/method.png with your method figure -->
![method](figure_method.pdf)

## 📰 News
- **[2026.06.12]** *EffiCell-Seg* has been accepted by MICCAI 2026 !
- **[2026.02.22]** We have released the code for *EffiCell-Seg* !

## ✨ Overview
*EffiCell-Seg* is a prompt-guided framework for cell instance segmentation built on a
frozen **DINOv3** backbone. Class-token similarity and PCA of the patch features form a
self-prompt that drives a SAM-style prompt encoder and a two-way transformer. The decoder
has two branches: a **binary** foreground branch and a **geometric** branch, fused by
cross-attention, and instances are recovered from the geometric output.

The geometric branch and its post-processing are **switchable** between four paradigms,
so the same architecture can be trained as any of:

| Paradigm   | Geometric target (channels)        | Instance decoding                         |
| ---------- | ---------------------------------- | ----------------------------------------- |
| `tsfd`     | normalized distance map (1)        | peak detection + watershed                |
| `hovernet` | horizontal / vertical maps (2)     | gradient maps + watershed                 |
| `stardist` | radial distances (`n_rays`)        | star-convex polygons + greedy NMS         |
| `cellpose` | flow field, dy/dx (2)              | flow tracking + sink clustering           |

Two running modes are supported:
- **`online`** — run the DINOv3 backbone on the fly (optionally with LoRA / CLS-token PEFT).
- **`cached`** — precompute DINOv3 features once with `extract_features.py`, then train only
  the decoder. Recommended for very large backbones (e.g. the 7B DINOv3) since the backbone
  runs only a single time over the dataset.

## 🛠 Setup
```bash
git clone <GITHUB_URL>
cd EffiCell-Seg

conda create -n efficellseg python=3.10 -y
conda activate efficellseg

# Core dependencies
pip install torch torchvision            # PyTorch 2.1+ (CUDA 12.x recommended)
pip install timm peft monai              # DINOv3 (timm), LoRA (peft), losses/metrics (monai)
pip install opencv-python scikit-image scipy pandas numpy tqdm


```
**Key requirements**: CUDA 12.x, PyTorch 2.1+, a recent `timm` that ships the DINOv3 weights.

> `eval.py` imports `get_fast_pq`, `get_fast_aji`, `remap_label` from `metrics`.
> If they are unavailable, instance-level metrics (PQ/SQ/DQ/AJI) fall back to 0 while the
> binary metrics (Dice/IoU/NSD/HD95) are still computed.

## 📚 Data Preparation
Each dataset lives under `./datasets/<DATASET>/processed`:
```
EffiCell-Seg
├── datasets
│   └── <DATASET>
│       └── processed
│           ├── images
│           │   ├── 0001.png
│           │   ├── ...
│           ├── npy                 # instance label maps (int), one .npy per image
│           │   ├── 0001.npy
│           │   ├── ...
│           └── ids.json
```
`ids.json` lists the split membership (file names or bare IDs):
```json
{
  "train": ["0001.png", "..."],
  "valid": ["0002.png", "..."],
  "test":  ["0003.png", "..."]
}
```
> `train.py` uses the `train` split for training and the `test` split for validation.
> `extract_features.py` processes **every** split key present in `ids.json`.

(Optional) precomputed features for `cached` mode are written to
`./precomputed_feats/<DATASET>/<split>/<id>.pt`.

## 🚀 Quickstart

### 1. (Optional) Precompute DINOv3 features — for `cached` mode
```bash
python extract_features.py \
  --input_dir ./datasets/<DATASET>/processed \
  --json_path ./datasets/<DATASET>/processed/ids.json \
  --output_dir ./precomputed_feats/<DATASET> \
  --model_name vit_7b_patch16_dinov3.lvd1689m \
  --img_size 512 --batch_size 4 --fp16
```

### 2. Train
Training uses `DistributedDataParallel`, so launch it with `torchrun` (single GPU works too):
```bash
# Single GPU, online mode, TSFD paradigm
torchrun --nproc_per_node=1 train.py \
  --dataset <DATASET> --mode online \
  --paradigm tsfd \
  --backbone timm/vit_7b_patch16_dinov3.lvd1689m \
  --img_size 512 --batch 4 --epoch 150 --lr 1e-4 \
  --output_dir outputs/efficellseg_<DATASET>
```
```bash
# Multi-GPU (e.g. 4 GPUs), cached mode, Cellpose paradigm
torchrun --nproc_per_node=4 train.py \
  --dataset <DATASET> --mode cached \
  --feature_dir ./precomputed_feats \
  --paradigm cellpose \
  --output_dir outputs/efficellseg_<DATASET>
```
Useful flags: `--paradigm {tsfd,hovernet,stardist,cellpose}`, `--n_rays` (stardist),
`--geo_weight` (override the geometric-loss weight), `--peft_strategy {lora,cls_only}`,
`--lora_rank`, `--num_workers`, `--sync_bn`. Checkpoints `best_loss_model.pth`,
`best_dice_model.pth`, and `latest_model.pth` are saved to `--output_dir`.

### 3. Evaluate
```bash
python eval.py \
  --dataset <DATASET> --mode online \
  --paradigm tsfd \
  --backbone vit_7b_patch16_dinov3.lvd1689m \
  --model_path outputs/efficellseg_<DATASET>/best_loss_model.pth \
  --post_processing instance \
  --output_dir visual_test/efficellseg_<DATASET>
```
- `--paradigm` **must match** the paradigm the checkpoint was trained with.
- `--post_processing instance` uses the paradigm-aware decoder; `simple` labels the binary
  map with connected components.
- Per-image metrics and visualizations (instance maps, boundaries, geometric maps, prompt /
  similarity / PCA maps) are written under `--output_dir`.

## 📊 Metrics
`eval.py` reports **Dice, IoU, NSD, PQ, SQ, DQ, F1, Precision, Recall, AJI, HD95**, saving a
per-image CSV and a mean/std summary.

## 📂 Repository Structure
```
EffiCell-Seg
├── model.py             # EffiCell-Seg model: DINOv3 wrapper, self-prompt, dual-branch decoder, per-paradigm post-processing
├── utils.py             # shared building blocks (LayerNorm2d, MLP, CrossAttention) + base class
├── dataloader.py        # dataset + paradigm-specific geometric-target generation
├── train.py             # DDP training with paradigm-aware losses
├── eval.py              # inference, metrics, and visualization
└── extract_features.py  # optional DINOv3 feature precomputation (for cached mode)
```

## 📜 Citation
If you find this work helpful, please consider citing:


## 🙏 Acknowledgements
- [DINOv3](https://github.com/facebookresearch/dinov3) / [timm](https://github.com/huggingface/pytorch-image-models)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [CellViT](https://github.com/TIO-IKIM/CellViT)
- [MONAI](https://github.com/Project-MONAI/MONAI)
- Instance-segmentation paradigms: [HoVer-Net](https://github.com/vqdang/hover_net), [StarDist](https://github.com/stardist/stardist), [Cellpose](https://github.com/MouseLand/cellpose), and TSFD-Net.
