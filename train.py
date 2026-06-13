import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import argparse
import os
import logging
import csv
import json
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.modules.loss import _Loss
from typing import Tuple

from dataloader import UnifiedCellDataset
from model import EffiCellSeg
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

# Default geometric-loss weights per paradigm (tunable; override with --geo_weight).
DEFAULT_GEO_WEIGHT = {'tsfd': 50.0, 'hovernet': 2.0, 'stardist': 0.3, 'cellpose': 5.0}

# =========================================================================
# Distributed / logging helpers
# =========================================================================

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend="nccl")
        rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, int(os.environ["WORLD_SIZE"])
    return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()

def reduce_tensor(tensor):
    if not dist.is_initialized(): return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / dist.get_world_size()

def setup_logging(save_prefix, log_dir='logs'):
    if dist.is_initialized() and dist.get_rank() != 0: return None
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(log_dir, f'{save_prefix}_{timestamp}.log')), logging.StreamHandler()])
    csv_file = os.path.join(log_dir, f'{save_prefix}_{timestamp}.csv')
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'phase', 'loss_total', 'loss_geo', 'loss_binary', 'dice_binary', 'lr'])
    return csv_file

# =========================================================================
# Trainer
# =========================================================================

class Trainer:
    def __init__(self, model, optimizer, scheduler, args, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda')

        self.paradigm = args.paradigm
        self.geo_weight = args.geo_weight
        self._sobel = None  # cached HoVer-Net gradient kernels

        pos_weight = torch.tensor([5.0]).to(device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss(sigmoid=True, softmax=False, batch=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

        # Geometric regression loss (used directly by tsfd; combined for hovernet).
        self.mse_loss = nn.MSELoss()

    def _get_sobel(self, device):
        """5x5 HoVer-Net gradient kernels: kh = h/(h^2+v^2), kv = v/(h^2+v^2)."""
        if self._sobel is not None and self._sobel[0].device == device:
            return self._sobel
        size = 5
        rng = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        v, h = torch.meshgrid(rng, rng, indexing='ij')  # h varies over cols, v over rows
        denom = (h * h + v * v) + 1e-15
        kh = (h / denom).view(1, 1, size, size).to(device)
        kv = (v / denom).view(1, 1, size, size).to(device)
        self._sobel = (kh, kv)
        return self._sobel

    def _msge_loss(self, pred, target, fg_mask):
        """HoVer-Net mean-squared gradient error over the foreground region.
        Gradient of the horizontal map along x and of the vertical map along y."""
        kh, kv = self._get_sobel(pred.device)

        def grad_hv(hv):
            dh = F.conv2d(hv[:, 0:1], kh, padding=2)
            dv = F.conv2d(hv[:, 1:2], kv, padding=2)
            return torch.cat([dh, dv], dim=1)

        diff = grad_hv(pred) - grad_hv(target)
        return (fg_mask * diff * diff).sum() / (fg_mask.sum() + 1e-8)

    def compute_geo_loss(self, pred_geo, geo_target, fg_mask):
        """Paradigm-specific geometric loss.
        - hovernet: MSE on the H/V maps + masked gradient (MSGE) term.
        - stardist: masked L1 on radial distances, summed over rays.
        - cellpose: MSE on the (dy, dx) flow field (background target is 0).
        - tsfd:     MSE on the distance map.
        """
        if self.paradigm == 'hovernet':
            return self.mse_loss(pred_geo, geo_target) + self._msge_loss(pred_geo, geo_target, fg_mask)
        if self.paradigm == 'stardist':
            l1 = (torch.abs(pred_geo - geo_target) * fg_mask).sum()
            return l1 / (fg_mask.sum() + 1e-8)
        return self.mse_loss(pred_geo, geo_target)

    def run_epoch(self, dataloader, phase='train', epoch=0):
        if phase == 'train':
            self.model.train()
            if isinstance(dataloader.sampler, DistributedSampler): dataloader.sampler.set_epoch(epoch)
        else:
            self.model.eval()

        self.dice_metric.reset()
        stats = {'loss_total': 0.0, 'loss_geo': 0.0, 'loss_binary': 0.0, 'dice_binary': 0.0}

        if len(dataloader) == 0: return stats

        pbar = tqdm(dataloader, desc=f'{phase} Ep {epoch}', disable=(dist.is_initialized() and dist.get_rank() != 0))

        for batch in pbar:
            inputs = {}
            if self.args.mode == 'cached':
                inputs['spatial_features'] = batch['spatial_features'].to(self.device, non_blocking=True)
                inputs['cls_token'] = batch['cls_token'].to(self.device, non_blocking=True)
            else:
                inputs['image'] = batch['image'].to(self.device, non_blocking=True)

            binary_map = batch['binary_map'].to(self.device, non_blocking=True).unsqueeze(1)
            geo_target = batch['geo_map'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                with torch.amp.autocast('cuda', enabled=True):
                    outputs = self.model(inputs)
                    pred_geo = outputs["geo_map"]
                    pred_binary_logits = outputs["nuclei_binary_map"]

                    # Geometric loss depends on the active paradigm.
                    loss_geo = self.compute_geo_loss(pred_geo, geo_target, binary_map)

                    # Binary foreground loss (BCE + Dice).
                    l_bce = self.bce(pred_binary_logits, binary_map)
                    l_dice = self.dice_loss(pred_binary_logits, binary_map)
                    loss_binary = l_bce + l_dice

                    # Weighted sum; geo_weight balances the two heads (tunable).
                    loss = loss_binary + self.geo_weight * loss_geo

                    if phase == 'train':
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                pred_mask = (pred_binary_logits > 0).float()
                self.dice_metric(y_pred=pred_mask, y=binary_map)

                r_loss = reduce_tensor(loss.detach())
                r_geo = reduce_tensor(loss_geo.detach())
                r_bin = reduce_tensor(loss_binary.detach())

                stats['loss_total'] += r_loss.item()
                stats['loss_geo'] += r_geo.item()
                stats['loss_binary'] += r_bin.item()

            pbar.set_postfix({'L': f'{r_loss.item():.3f}', 'Geo': f'{r_geo.item():.3f}', 'Bin': f'{r_bin.item():.3f}'})

        for k in stats: stats[k] /= len(dataloader)
        try:
            dice_val = self.dice_metric.aggregate().item()
            stats['dice_binary'] = reduce_tensor(torch.tensor(dice_val).cuda()).item()
        except Exception:
            stats['dice_binary'] = 0.0

        return stats

# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nips22')
    parser.add_argument('--feature_dir', type=str, default='./precomputed_feats')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--output_dir', type=str, default='outputs/efficellseg_nips')
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--decoder_dim', type=int, default=256)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--filter_empty', action='store_true')
    # Instance-seg paradigm and its options.
    parser.add_argument('--paradigm', type=str, default='tsfd', choices=['tsfd', 'hovernet', 'stardist', 'cellpose'])
    parser.add_argument('--n_rays', type=int, default=32, help='Number of radial directions (stardist only).')
    parser.add_argument('--geo_weight', type=float, default=None,
                        help='Weight of the geometric loss. Defaults per paradigm if unset.')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--mode', type=str, default='online', choices=['cached', 'online'])
    parser.add_argument('--backbone', type=str, default='timm/vit_7b_patch16_dinov3.lvd1689m')
    parser.add_argument('--peft', default=True, action='store_true')
    parser.add_argument('--peft_strategy', type=str, default='cls_only', choices=['lora', 'cls_only'])
    parser.add_argument('--lora_rank', type=int, default=16)
    # Enable SyncBatchNorm (off by default; can slow training down).
    parser.add_argument('--sync_bn', action='store_true', help='Enable SyncBatchNorm (may cause lag)')
    # Number of dataloader workers.
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Resolve the geometric-loss weight from the paradigm default if not provided.
    if args.geo_weight is None:
        args.geo_weight = DEFAULT_GEO_WEIGHT[args.paradigm]

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        csv_filename = setup_logging(f"{args.dataset}_{args.paradigm}_{args.mode}", args.output_dir)
    else:
        csv_filename = None

    split_path = f"./datasets/{args.dataset}/processed/ids.json"
    with open(split_path, 'r') as f: ids_data = json.load(f)

    train_root = os.path.join(args.feature_dir, args.dataset, 'train') if args.mode == 'cached' else None
    val_root = os.path.join(args.feature_dir, args.dataset, 'test') if args.mode == 'cached' else None

    train_ds = UnifiedCellDataset(args.dataset, ids_data['train'], train_root, phase='train', mode=args.mode,
                                  filter_empty=args.filter_empty, paradigm=args.paradigm, n_rays=args.n_rays,
                                  img_size=args.img_size)
    val_ds = UnifiedCellDataset(args.dataset, ids_data['test'], val_root, phase='valid', mode=args.mode,
                                paradigm=args.paradigm, n_rays=args.n_rays, img_size=args.img_size)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    # num_workers must be > 0 to use prefetch_factor.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=2)

    model = EffiCellSeg(
        mode=args.mode, backbone_name=args.backbone, use_peft=args.peft,
        peft_strategy=args.peft_strategy, lora_rank=args.lora_rank,
        embed_dim=args.embed_dim, decoder_dim=args.decoder_dim,
        paradigm=args.paradigm, n_rays=args.n_rays,
        img_size=args.img_size
    )
    model = model.to(device)

    # Only convert to SyncBatchNorm when explicitly requested.
    if args.sync_bn:
        if rank == 0: logging.info("Using SyncBatchNorm (Warning: might be slow)")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    trainer = Trainer(model, optimizer, scheduler, args, device)

    # ==========================
    # Best checkpoint trackers
    # ==========================
    best_dice = 0.0
    best_loss = float('inf')

    for epoch in range(args.epoch):
        t_stats = trainer.run_epoch(train_loader, 'train', epoch)
        v_stats = trainer.run_epoch(val_loader, 'valid', epoch)
        scheduler.step()

        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            with open(csv_filename, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, 'train', t_stats['loss_total'], t_stats['loss_geo'], t_stats['loss_binary'], t_stats['dice_binary'], current_lr])
                csv.writer(f).writerow([epoch, 'valid', v_stats['loss_total'], v_stats['loss_geo'], v_stats['loss_binary'], v_stats['dice_binary'], current_lr])

            logging.info(f"Ep {epoch} | T_Loss: {t_stats['loss_total']:.4f} Dice: {t_stats['dice_binary']:.3f} | V_Loss: {v_stats['loss_total']:.4f} Dice: {v_stats['dice_binary']:.3f}")

            if v_stats['loss_total'] < best_loss:
                best_loss = v_stats['loss_total']
                torch.save(model.module.state_dict(), os.path.join(args.output_dir, "best_loss_model.pth"))
                logging.info(f"==> New best loss: {best_loss:.4f}, saved best_loss_model.pth")

            if v_stats['dice_binary'] > best_dice:
                best_dice = v_stats['dice_binary']
                torch.save(model.module.state_dict(), os.path.join(args.output_dir, "best_dice_model.pth"))
                logging.info(f"==> New best dice: {best_dice:.4f}, saved best_dice_model.pth")

            torch.save(model.module.state_dict(), os.path.join(args.output_dir, "latest_model.pth"))

    cleanup_ddp()
