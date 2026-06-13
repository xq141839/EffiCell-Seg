import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pandas as pd
import cv2
import os
import json
from tqdm import tqdm
from skimage.metrics import hausdorff_distance
from monai.metrics import DiceMetric
from typing import List
from scipy.ndimage import find_objects, distance_transform_edt
from skimage.segmentation import find_boundaries
from skimage.measure import label as sk_label

# Model and dataloader.
from model import EffiCellSeg
from dataloader import UnifiedCellDataset

# Instance-segmentation metrics.
try:
    from metrics import get_fast_pq, remap_label, get_fast_aji
except ImportError:
    print("Warning: Could not import 'sam2.utils.metrics'. Please ensure your environment is configured correctly.")
    def get_fast_pq(t, p): return 0, 0, 0
    def get_fast_aji(t, p): return 0
    def remap_label(m): return m

# =========================================================
# Metric helpers (matching reference)
# =========================================================

def cell_detection_scores(paired_true, paired_pred, unpaired_true, unpaired_pred, w: List = [1, 1]):
    if hasattr(paired_pred, '__len__'): tp_d = len(paired_pred)
    else: tp_d = paired_pred

    if hasattr(unpaired_pred, '__len__'): fp_d = len(unpaired_pred)
    else: fp_d = unpaired_pred

    if hasattr(unpaired_true, '__len__'): fn_d = len(unpaired_true)
    else: fn_d = unpaired_true

    prec_d = tp_d / (tp_d + fp_d) if tp_d + fp_d > 0 else 0.0
    rec_d = tp_d / (tp_d + fn_d) if tp_d + fn_d > 0 else 0.0
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d) if 2 * tp_d + w[0] * fp_d + w[1] * fn_d > 0 else 0.0

    return f1_d, prec_d, rec_d

def compute_tp_fp_fn(gt_map, pred_map):
    gt_map = remap_label(gt_map)
    pred_map = remap_label(pred_map)

    true_ids = np.unique(gt_map)
    pred_ids = np.unique(pred_map)
    true_ids = true_ids[true_ids > 0]
    pred_ids = pred_ids[pred_ids > 0]

    if len(true_ids) == 0 and len(pred_ids) == 0: return 0, 0, 0
    if len(true_ids) == 0: return 0, len(pred_ids), 0
    if len(pred_ids) == 0: return 0, 0, len(true_ids)

    gt_areas = np.bincount(gt_map.ravel())
    pred_areas = np.bincount(pred_map.ravel())

    tp = 0
    slices = find_objects(gt_map)
    for idx, sl in enumerate(slices):
        if sl is None: continue
        gid = idx + 1
        if gid >= len(gt_areas): continue

        gt_crop = gt_map[sl]
        pred_crop = pred_map[sl]
        g_mask = (gt_crop == gid)
        coinciding_preds = pred_crop[g_mask]
        coinciding_preds = coinciding_preds[coinciding_preds > 0]

        if len(coinciding_preds) == 0: continue
        p_counts = np.bincount(coinciding_preds)
        pid = np.argmax(p_counts)
        intersection = p_counts[pid]

        union = gt_areas[gid] + pred_areas[pid] - intersection
        if intersection / union > 0.5:
            tp += 1

    return tp, len(pred_ids) - tp, len(true_ids) - tp

def compute_binary_nsd(pred, gt, tau=2.0):
    """Normalized Surface Distance for binary masks.
    pred, gt: binary maps (0/1). tau: tolerance in pixels."""
    if pred.sum() == 0 and gt.sum() == 0: return 1.0
    if pred.sum() == 0 or gt.sum() == 0: return 0.0

    # Inner-mode boundaries keep the border inside the foreground.
    pred_border = find_boundaries(pred, mode='inner')
    gt_border = find_boundaries(gt, mode='inner')

    # distance_transform_edt measures distance to non-zero points, so feed ~border.
    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    # Boundary pixels falling within the tolerance of the other boundary.
    pred_in_tol = np.sum(pred_border & (dt_gt <= tau))
    gt_in_tol = np.sum(gt_border & (dt_pred <= tau))

    denom = np.sum(pred_border) + np.sum(gt_border)
    if denom == 0: return 1.0

    return (pred_in_tol + gt_in_tol) / denom

# =========================================================
# Visualization helpers
# =========================================================

def colorize_instance_map(instance_map):
    if np.max(instance_map) == 0: return np.zeros((instance_map.shape[0], instance_map.shape[1], 3), dtype=np.uint8)
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, size=(np.max(instance_map) + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors[instance_map]

def extract_boundary_vis(instance_map):
    boundaries = find_boundaries(instance_map, mode='inner')
    boundaries_uint8 = (boundaries.astype(np.uint8) * 255)
    return cv2.cvtColor(boundaries_uint8, cv2.COLOR_GRAY2BGR)

def get_iou(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0: return 1.0
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    return intersection / (union + 1e-6)

# =========================================================
# Main inference logic
# =========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nips22', type=str)
    parser.add_argument('--feature_dir', type=str, default='./precomputed_feats', help='Used if mode=cached')
    parser.add_argument('--model_path', default='outputs/efficellseg_nips/best_loss_model.pth', type=str)
    parser.add_argument('--output_dir', type=str, default='visual_test/efficellseg_results')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--backbone', type=str, default='vit_7b_patch16_dinov3.lvd1689m')
    parser.add_argument('--mode', type=str, default='online', choices=['cached', 'online'])
    parser.add_argument('--paradigm', type=str, default='tsfd', choices=['tsfd', 'hovernet', 'stardist', 'cellpose'],
                        help='Must match the paradigm the checkpoint was trained with.')
    parser.add_argument('--n_rays', type=int, default=32, help='Number of radial directions (stardist only).')
    parser.add_argument('--post_processing', type=str, default='simple', choices=['simple', 'instance'],
                        help="'simple' (binary + connected components) or 'instance' (paradigm-aware decoding)")
    parser.add_argument('--nsd_tau', type=float, default=2.0, help='Tolerance for NSD calculation')

    args = parser.parse_args()

    vis_root = os.path.join(args.output_dir, args.dataset, 'vis')
    os.makedirs(vis_root, exist_ok=True)

    split_path = f"./datasets/{args.dataset}/processed/ids.json"
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, 'r') as f: ids_data = json.load(f)
    test_files = ids_data['test']
    print(f"Loaded {len(test_files)} test files from {args.dataset}")

    val_root = os.path.join(args.feature_dir, args.dataset, 'test') if args.mode == 'cached' else None

    test_dataset = UnifiedCellDataset(
        data_name=args.dataset,
        jsfiles=test_files,
        root_dir=val_root,
        phase='test',
        mode=args.mode,
        img_size=args.img_size,
        paradigm=args.paradigm,
        n_rays=args.n_rays
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {args.model_path}")

    model = EffiCellSeg(
        mode=args.mode,
        backbone_name=args.backbone,
        img_size=args.img_size,
        decoder_dim=256,
        paradigm=args.paradigm,
        n_rays=args.n_rays,
        use_peft=False
    )

    state_dict = torch.load(args.model_path, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    metric_keys = ['Dice', 'IoU', 'NSD', 'PQ', 'SQ', 'DQ', 'F1', 'Precision', 'Recall', 'AJI', 'HD95']
    metrics = {k: [] for k in metric_keys}
    image_ids = []

    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    print(f"Starting inference (Post-processing: {args.post_processing})...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            img_id = batch["image_id"][0]

            inputs = {}
            if args.mode == 'cached':
                inputs['spatial_features'] = batch['spatial_features'].to(device)
                inputs['cls_token'] = batch['cls_token'].to(device)
            else:
                inputs['image'] = batch['image'].to(device)

            gt_inst = batch['inst_map'][0].numpy()
            gt_binary = batch['binary_map'][0].numpy()

            torch.cuda.synchronize()
            outputs = model(inputs)

            pred_binary_logits = outputs["nuclei_binary_map"]
            pred_binary = (torch.sigmoid(pred_binary_logits) > 0.5).cpu().numpy()[0, 0]

            if args.post_processing == 'simple':
                pred_inst = sk_label(pred_binary).astype(int)
            else:
                pred_inst_tensor, _ = model.calculate_instance_map(outputs)
                pred_inst = pred_inst_tensor[0].numpy().astype(int)

            gt_inst_remap = remap_label(gt_inst)
            pred_inst_remap = remap_label(pred_inst)

            pq, dq, sq, aji = 0.0, 0.0, 0.0, 0.0
            if gt_inst_remap.max() > 0 or pred_inst_remap.max() > 0:
                try:
                    pq_res = get_fast_pq(gt_inst_remap, pred_inst_remap)
                    aji = get_fast_aji(gt_inst_remap, pred_inst_remap)
                    if isinstance(pq_res, (list, tuple)) and len(pq_res) >= 3:
                        dq, sq, pq = pq_res[0], pq_res[1], pq_res[2]
                    elif isinstance(pq_res, (list, tuple)):
                        pq = pq_res[0]
                    else:
                        pq = pq_res
                except Exception: pass

            tp, fp, fn = compute_tp_fp_fn(gt_inst_remap, pred_inst_remap)
            f1, precision, recall = cell_detection_scores(tp, tp, fn, fp)

            dice = 2 * np.logical_and(pred_binary, gt_binary).sum() / (pred_binary.sum() + gt_binary.sum() + 1e-6)
            iou = get_iou(pred_binary, gt_binary)

            nsd = compute_binary_nsd(pred_binary, gt_binary, tau=args.nsd_tau)

            try:
                if pred_binary.sum() > 0 and gt_binary.sum() > 0:
                    hd95 = hausdorff_distance(pred_binary, gt_binary, percentile=95)
                else:
                    hd95 = np.nan
            except Exception:
                hd95 = np.nan

            metrics['Dice'].append(dice)
            metrics['IoU'].append(iou)
            metrics['NSD'].append(nsd)
            metrics['PQ'].append(pq)
            metrics['SQ'].append(sq)
            metrics['DQ'].append(dq)
            metrics['F1'].append(f1)
            metrics['Precision'].append(precision)
            metrics['Recall'].append(recall)
            metrics['AJI'].append(aji)
            metrics['HD95'].append(hd95)
            image_ids.append(img_id)

            # --- Visualization ---
            img_save_dir = os.path.join(vis_root, str(img_id))
            os.makedirs(img_save_dir, exist_ok=True)
            target_size = (args.img_size, args.img_size)

            # 1. Raw image
            if args.mode == 'online':
                img_tensor = batch['image'][0].cpu()
                img_vis = img_tensor * pixel_std + pixel_mean
                img_np = img_vis.clamp(0, 1).permute(1, 2, 0).numpy()
                img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(img_save_dir, "raw.png"), img_bgr)
            else:
                img_path_png = os.path.join(f'./datasets/{args.dataset}/processed/images', f"{img_id}.png")
                if os.path.exists(img_path_png):
                    raw_img = cv2.imread(img_path_png)
                    if raw_img.shape[0] != args.img_size:
                        raw_img = cv2.resize(raw_img, target_size)
                    cv2.imwrite(os.path.join(img_save_dir, "raw.png"), raw_img)

            # 2. Instance & boundary maps
            cv2.imwrite(os.path.join(img_save_dir, "pred_inst.png"), colorize_instance_map(pred_inst))
            cv2.imwrite(os.path.join(img_save_dir, "gt_inst.png"), colorize_instance_map(gt_inst))
            cv2.imwrite(os.path.join(img_save_dir, "pred_boundary.png"), extract_boundary_vis(pred_inst))
            cv2.imwrite(os.path.join(img_save_dir, "gt_boundary.png"), extract_boundary_vis(gt_inst))

            # 3. Binary & geometric maps
            pred_bin_uint8 = (pred_binary * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(img_save_dir, "pred_binary.png"), pred_bin_uint8)

            gt_bin_uint8 = (gt_binary * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(img_save_dir, "gt_binary.png"), gt_bin_uint8)

            # First geometric channel (distance for tsfd; horizontal map / first ray otherwise).
            pred_dist = outputs["geo_map"][0].cpu().numpy()[0]
            dist_norm = cv2.normalize(pred_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(img_save_dir, "pred_geo.png"), cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET))

            gt_dist = batch['geo_map'][0, 0].numpy()
            gt_dist_norm = cv2.normalize(gt_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(img_save_dir, "gt_geo.png"), cv2.applyColorMap(gt_dist_norm, cv2.COLORMAP_JET))

            # 4. Feature maps
            if "pca_map" in outputs:
                pca_vis = outputs["pca_map"][0].cpu().numpy().transpose(1, 2, 0)
                pca_vis = (pca_vis * 255).astype(np.uint8)
                pca_vis = cv2.resize(pca_vis, target_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(img_save_dir, "pca_map.png"), cv2.cvtColor(pca_vis, cv2.COLOR_RGB2BGR))

            if "sim_map" in outputs:
                sim_vis = outputs["sim_map"][0, 0].cpu().numpy()
                sim_vis = (sim_vis * 255).astype(np.uint8)
                sim_vis = cv2.resize(sim_vis, target_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(img_save_dir, "sim_map.png"), cv2.applyColorMap(sim_vis, cv2.COLORMAP_JET))

            if "prompt_map" in outputs:
                prompt_tensor = outputs["prompt_map"][0]

                # Raw RGB prompt map (PCA * similarity).
                prompt_vis = prompt_tensor.cpu().numpy().transpose(1, 2, 0)
                prompt_vis = (prompt_vis * 255).astype(np.uint8)
                prompt_vis = cv2.resize(prompt_vis, target_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(img_save_dir, "prompt_map.png"), cv2.cvtColor(prompt_vis, cv2.COLOR_RGB2BGR))

                # Saliency map (per-pixel L2 norm aggregated to one channel).
                saliency_map = torch.norm(prompt_tensor, p=2, dim=0)  # [Hp, Wp]
                saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6)
                saliency_vis = (saliency_map.cpu().numpy() * 255).astype(np.uint8)
                saliency_vis = cv2.resize(saliency_vis, target_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(img_save_dir, "saliency_map.png"), cv2.applyColorMap(saliency_vis, cv2.COLORMAP_JET))

    # --- Summary ---
    df = pd.DataFrame(metrics)
    df.insert(0, 'Image ID', image_ids)

    summary = df.describe().loc[['mean', 'std']]
    print("\n=== Evaluation Summary ===")
    print(summary)

    csv_path = os.path.join(args.output_dir, f"{args.dataset}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to {csv_path}")

    summary_path = os.path.join(args.output_dir, f"{args.dataset}_summary.csv")
    summary.to_csv(summary_path)
