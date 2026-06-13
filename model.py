import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model
# Shared building blocks live in utils.py. BaseCellSeg is the renamed base class
# (formerly named after a third-party method) that this model subclasses.
from utils import BaseCellSeg, MLP, CrossAttentionModule, LayerNorm2d
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.prompt_encoder import PromptEncoder
from typing import Tuple, List, Literal
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label as sk_label
from skimage.morphology import remove_small_objects, disk
from scipy.ndimage import binary_fill_holes, distance_transform_edt
import numpy as np
import cv2

# Geometric-head output channels per paradigm. 'stardist' is resolved to n_rays at init.
GEO_CHANNELS = {"tsfd": 1, "hovernet": 2, "stardist": None, "cellpose": 2}


# === Backbone Wrapper ===
class DinoV3BackboneWrapper(nn.Module):
    def __init__(self, model_name, pretrained=True, img_size=512):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True
        )
        if hasattr(self.model, 'patch_embed'):
            self.patch_size = self.model.patch_embed.patch_size[0]
        else:
            self.patch_size = 14

        self.embed_dim = self.model.embed_dim
        self.is_lora = False

    def apply_peft(self, strategy: Literal['lora', 'cls_only'] = 'lora', rank=16):
        for param in self.model.parameters():
            param.requires_grad = False

        if strategy == 'lora':
            print(f"[Model] Applying LoRA (Rank={rank}) to Backbone...")
            config = LoraConfig(
                r=rank, lora_alpha=rank * 2,
                target_modules=["qkv", "proj", "fc1", "fc2", "q_proj", "v_proj"],
                lora_dropout=0.05, bias="none", modules_to_save=[],
            )
            self.model = get_peft_model(self.model, config)
            self.is_lora = True
            self.model.print_trainable_parameters()

        elif strategy == 'cls_only':
            print("[Model] PEFT Strategy: CLS Token Tuning (Backbone frozen, CLS unmasked).")
            if hasattr(self.model, 'cls_token'):
                self.model.cls_token.requires_grad = True
                print("Enabled gradients for cls_token.")
            else:
                print("Warning: cls_token not found in this model architecture.")
            self.is_lora = False

    def forward(self, x):
        B, C, H, W = x.shape
        model_obj = self.model.base_model.model if self.is_lora else self.model

        if hasattr(model_obj, 'forward_features'):
            x_feats = model_obj.forward_features(x)
        else:
            x_feats = model_obj(x)

        n_prefix = getattr(model_obj, 'num_prefix_tokens', 1)

        patch_tokens = x_feats[:, n_prefix:, :] if n_prefix > 0 else x_feats
        if n_prefix > 0:
            cls_token = x_feats[:, 0, :]
        else:
            cls_token = x_feats.mean(dim=1)

        h_grid = H // self.patch_size
        w_grid = W // self.patch_size

        patch_tokens = patch_tokens.reshape(B, h_grid, w_grid, self.embed_dim)
        spatial_features = patch_tokens.permute(0, 3, 1, 2).contiguous()

        return spatial_features, cls_token


# === EffiCell-Seg Model ===
class EffiCellSeg(BaseCellSeg):
    """EffiCell-Seg: prompt-guided cell instance segmentation with a switchable
    decoding paradigm (tsfd / hovernet / stardist)."""

    def __init__(
        self,
        mode: Literal['cached', 'online'] = 'cached',
        backbone_name: str = 'vit_large_patch14_dinov2.lvd142m',
        use_peft: bool = False,
        peft_strategy: Literal['lora', 'cls_only'] = 'cls_only',
        lora_rank: int = 16,
        embed_dim: int = 1024,
        decoder_dim: int = 256,
        paradigm: Literal['tsfd', 'hovernet', 'stardist', 'cellpose'] = 'tsfd',
        n_rays: int = 32,
        drop_rate: float = 0,
        img_size: int = 512,
    ):
        super().__init__(2, 1, embed_dim, 3, 24, 16, [], 0.25, True, drop_rate, False)

        if paradigm not in GEO_CHANNELS:
            raise ValueError(f"Unknown paradigm '{paradigm}'. Choose from {list(GEO_CHANNELS)}.")

        self.mode = mode
        self.decoder_dim = decoder_dim
        self.paradigm = paradigm
        self.n_rays = n_rays
        self.img_size = img_size

        if self.mode == 'online':
            self.backbone = DinoV3BackboneWrapper(model_name=backbone_name, img_size=img_size)
            if use_peft:
                self.backbone.apply_peft(strategy=peft_strategy, rank=lora_rank)
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            embed_dim = self.backbone.embed_dim

        # Geometric channels: tsfd -> 1 (distance), hovernet -> 2 (H/V), stardist -> n_rays.
        self.geo_channels = n_rays if paradigm == 'stardist' else GEO_CHANNELS[paradigm]

        self.projection = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_dim, kernel_size=1, bias=False), LayerNorm2d(decoder_dim),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False), LayerNorm2d(decoder_dim),
        )

        self.prompt_encoder = PromptEncoder(
            embed_dim=decoder_dim,
            image_embedding_size=(img_size // self.patch_size, img_size // self.patch_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16
        )

        self.prompt_encoder.no_mask_embed.weight.requires_grad = False
        for layer in self.prompt_encoder.point_embeddings:
            layer.weight.requires_grad = False
        self.prompt_encoder.not_a_point_embed.weight.requires_grad = False

        hyper_hidden = decoder_dim

        # Segmentation branch (binary foreground).
        self.transformer_seg = TwoWayTransformer(depth=2, embedding_dim=decoder_dim, mlp_dim=decoder_dim, num_heads=8)
        self.mask_tokens_seg = nn.Embedding(2, decoder_dim)
        self.hyper_mlps_seg = nn.ModuleList([MLP(decoder_dim, hyper_hidden, 16, 3) for _ in range(2)])

        # Geometric branch (paradigm-specific maps).
        self.transformer_geo = TwoWayTransformer(depth=2, embedding_dim=decoder_dim, mlp_dim=decoder_dim, num_heads=8)
        self.mask_tokens_geo = nn.Embedding(self.geo_channels, decoder_dim)
        self.hyper_mlps_geo = nn.ModuleList([MLP(decoder_dim, hyper_hidden, 16, 3) for _ in range(self.geo_channels)])

        self.geo_token = nn.Embedding(1, decoder_dim)
        self.seg_token = nn.Embedding(1, decoder_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, 64, kernel_size=2, stride=2), LayerNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), LayerNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.GELU(),
        )
        self.cross_attention = CrossAttentionModule(dim=decoder_dim, num_heads=8)

    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        print(f"--- Model Parameter Summary ---")
        print(f"Trainable params: {trainable_params:,d}")
        print(f"All params:       {all_params:,d}")
        print(f"Trainable%:       {100 * trainable_params / all_params:.4f}%")
        print(f"-------------------------------")

    def compute_pca_sim_prompt(self, spatial_features, cls_token):
        B, C, Hp, Wp = spatial_features.shape
        tokens = spatial_features.flatten(2).transpose(1, 2)

        spatial_norm = F.normalize(tokens, p=2, dim=2)
        cls_norm = F.normalize(cls_token, p=2, dim=1).unsqueeze(1)
        sim = torch.bmm(spatial_norm, cls_norm.transpose(1, 2))
        sim_map = ((sim + 1.0) / 2.0).transpose(1, 2).view(B, 1, Hp, Wp)

        pca_maps = []
        for i in range(B):
            x = tokens[i]
            x_mean = x.mean(dim=0, keepdim=True)
            x_centered = x - x_mean
            try:
                _, _, V = torch.pca_lowrank(x_centered, q=3, center=False, niter=3)
                pca_proj = torch.matmul(x_centered, V[:, :3])
            except Exception:
                U, S, V = torch.svd(x_centered)
                pca_proj = torch.matmul(x_centered, V[:, :3])

            pca_min = pca_proj.min(dim=0, keepdim=True)[0]
            pca_max = pca_proj.max(dim=0, keepdim=True)[0]
            pca_norm = (pca_proj - pca_min) / (pca_max - pca_min + 1e-6)

            pca_maps.append(pca_norm.transpose(0, 1).view(3, Hp, Wp))

        pca_map = torch.stack(pca_maps)
        prompt_map = pca_map * sim_map

        return prompt_map, sim_map, pca_map

    def predict_from_interactive_features(self, feat, hs, mlps, output_size):
        B, C, H, W = feat.shape
        upscaled = self.output_upscaling(feat)
        mask_tokens_out = hs[:, 1: (1 + len(mlps)), :]
        hyper_in_list = [mlps[i](mask_tokens_out[:, i, :]) for i in range(len(mlps))]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
        return F.interpolate(masks, size=output_size, mode="bilinear", align_corners=False)

    # === Paradigm-specific post-processing ===

    def post_process_tsfd(self, dist_map, prob_map):
        """TSFD-style: find peaks on the distance map, then watershed."""
        dist_np = dist_map.squeeze().cpu().numpy()
        prob_np = prob_map.cpu().numpy()

        mask = (prob_np > 0.5)
        dist_np = dist_np * mask

        coords = peak_local_max(dist_np, min_distance=6, threshold_abs=0.4, labels=mask)
        markers = np.zeros_like(dist_np, dtype=int)
        markers[tuple(coords.T)] = np.arange(len(coords)) + 1

        inst_map = watershed(-dist_np, markers, mask=mask)
        return inst_map

    def post_process_hovernet(self, hv_map, prob_map, min_size=10):
        """HoVer-Net style: gradient of the H/V maps gives boundaries, then watershed."""
        prob = prob_map.cpu().numpy()
        h_dir_raw = hv_map[0].cpu().numpy()
        v_dir_raw = hv_map[1].cpu().numpy()

        blb = (prob >= 0.5).astype(np.int32)
        blb = sk_label(blb)
        blb = remove_small_objects(blb, min_size=min_size)
        blb[blb > 0] = 1
        blb = blb.astype(np.int32)

        h_dir = cv2.normalize(h_dir_raw, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        v_dir = cv2.normalize(v_dir_raw, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
        sobelh = 1 - cv2.normalize(sobelh, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        sobelv = 1 - cv2.normalize(sobelv, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = (overall >= 0.4).astype(np.int32)
        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype(np.uint8)
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, disk(2).astype(np.uint8))
        marker = sk_label(marker)
        marker = remove_small_objects(marker, min_size=min_size)

        inst_map = watershed(dist, markers=marker, mask=blb)
        return inst_map

    def post_process_stardist(self, dist_map, prob_map, prob_thresh=0.5, nms_thresh=0.4):
        """StarDist style: reconstruct star-convex polygons from radial distances,
        then greedy overlap-based NMS. Dependency-free decoder; swap in stardist's
        non_maximum_suppression if exact parity is needed."""
        dist = dist_map.cpu().numpy()
        prob = prob_map.cpu().numpy()
        n_rays, H, W = dist.shape

        phis = 2.0 * np.pi * np.arange(n_rays) / n_rays
        sin_p, cos_p = np.sin(phis), np.cos(phis)

        fg = prob > prob_thresh
        coords = peak_local_max(prob, min_distance=3, threshold_abs=prob_thresh, labels=fg)
        if len(coords) == 0:
            return np.zeros((H, W), dtype=np.int32)

        scores = prob[coords[:, 0], coords[:, 1]]
        coords = coords[np.argsort(-scores)]

        inst_map = np.zeros((H, W), dtype=np.int32)
        occupied = np.zeros((H, W), dtype=bool)
        label = 0
        for r, c in coords:
            rr = dist[:, r, c]
            ys = np.clip(np.round(r + rr * sin_p), 0, H - 1).astype(np.int32)
            xs = np.clip(np.round(c + rr * cos_p), 0, W - 1).astype(np.int32)
            poly = np.stack([xs, ys], axis=1)  # cv2 expects (x, y)

            cand = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(cand, [poly], 1)
            cand = cand.astype(bool)
            area = cand.sum()
            if area == 0:
                continue
            if np.logical_and(cand, occupied).sum() / area > nms_thresh:
                continue

            label += 1
            inst_map[cand & (~occupied)] = label
            occupied |= cand
        return inst_map

    def post_process_cellpose(self, flow_map, prob_map, n_iter=200, prob_thresh=0.5,
                              min_seed_distance=5, min_seed_count=5):
        """Cellpose style: integrate pixels along the predicted flow field to their
        attractor (cell center) via Euler steps, histogram the sink locations to find
        seeds (cell centers), then label each foreground pixel by its nearest seed
        (Voronoi). Dependency-free decoder; swap in cellpose.dynamics for exact parity."""
        device = flow_map.device
        flows = flow_map.float()  # [2, H, W] = (dy, dx)
        H, W = flows.shape[1], flows.shape[2]

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        coords = torch.stack([ys, xs], dim=0).float()  # [2, H, W]

        flows_b = flows.unsqueeze(0)
        for _ in range(n_iter):
            grid = coords.permute(1, 2, 0).unsqueeze(0).clone()  # [1, H, W, 2] = (y, x)
            grid[..., 0] = 2.0 * grid[..., 0] / max(H - 1, 1) - 1.0
            grid[..., 1] = 2.0 * grid[..., 1] / max(W - 1, 1) - 1.0
            grid = torch.flip(grid, dims=[-1])  # grid_sample expects (x, y)
            step = F.grid_sample(flows_b, grid, mode='bilinear',
                                 padding_mode='border', align_corners=True).squeeze(0)
            coords = coords + step
            coords[0].clamp_(0, H - 1)
            coords[1].clamp_(0, W - 1)

        sink = torch.round(coords).long().cpu().numpy()  # [2, H, W]
        mask = (prob_map.float() > prob_thresh).cpu().numpy()
        inst = np.zeros((H, W), dtype=np.int32)
        if mask.sum() == 0:
            return inst

        sy = sink[0][mask]
        sx = sink[1][mask]

        # Sink histogram; peaks are cell centers.
        hist = np.zeros((H, W), dtype=np.float32)
        np.add.at(hist, (sy, sx), 1.0)
        seeds = peak_local_max(hist, min_distance=min_seed_distance, threshold_abs=min_seed_count)
        if len(seeds) == 0:
            return inst

        seed_map = np.zeros((H, W), dtype=np.int32)
        seed_map[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)

        # Nearest-seed label everywhere (Voronoi via EDT indices), then assign each
        # foreground pixel by the seed nearest to where its flow converged.
        _, (iy, ix) = distance_transform_edt(seed_map == 0, return_indices=True)
        nearest = seed_map[iy, ix]
        inst[mask] = nearest[sy, sx]
        return inst

    def calculate_instance_map(self, predictions: dict, magnification: int = 40) -> Tuple[torch.Tensor, List[dict]]:
        geo_map = predictions["geo_map"]
        binary_logits = predictions["nuclei_binary_map"]
        prob_maps = torch.sigmoid(binary_logits)[:, 0]

        instance_preds = []
        for i in range(geo_map.shape[0]):
            geo = geo_map[i]
            prob = prob_maps[i]
            if self.paradigm == 'hovernet':
                inst = self.post_process_hovernet(geo, prob)
            elif self.paradigm == 'stardist':
                inst = self.post_process_stardist(geo, prob)
            elif self.paradigm == 'cellpose':
                inst = self.post_process_cellpose(geo, prob)
            else:
                inst = self.post_process_tsfd(geo, prob)
            instance_preds.append(inst)

        type_preds = []
        for inst in instance_preds:
            t_pred = {}
            for uid in np.unique(inst):
                if uid == 0:
                    continue
                t_pred[uid] = {"type": 1}
            type_preds.append(t_pred)

        return torch.tensor(np.stack(instance_preds)), type_preds

    def forward(self, inputs: dict) -> dict:
        if self.mode == 'online':
            images = inputs['image']
            spatial, cls_tok = self.backbone(images)
        else:
            spatial = inputs['spatial_features']
            cls_tok = inputs['cls_token']

        B, C, Hp, Wp = spatial.shape
        H, W = self.img_size, self.img_size

        out_dict = {}
        img_emb = self.projection(spatial)

        prompt_map, sim_map, pca_map = self.compute_pca_sim_prompt(spatial, cls_tok)
        out_dict["sim_map"] = sim_map
        out_dict["pca_map"] = pca_map
        out_dict["prompt_map"] = prompt_map

        sim_prompt = prompt_map
        sparse_emb, dense_emb = self.prompt_encoder(points=None, boxes=None, masks=sim_prompt)

        image_pe = self.prompt_encoder.get_dense_pe()
        src_base = img_emb + dense_emb

        # Segmentation branch (binary).
        tokens_seg = torch.cat([self.seg_token.weight, self.mask_tokens_seg.weight], dim=0).unsqueeze(0).expand(B, -1, -1)
        hs_seg, up_src_seg = self.transformer_seg(src_base, image_pe, tokens_seg)
        feat_seg = up_src_seg.transpose(1, 2).view(B, self.decoder_dim, Hp, Wp)

        # Geometric branch (paradigm-specific).
        tokens_geo = torch.cat([self.geo_token.weight, self.mask_tokens_geo.weight], dim=0).unsqueeze(0).expand(B, -1, -1)
        hs_geo, up_src_geo = self.transformer_geo(src_base, image_pe, tokens_geo)
        feat_geo = up_src_geo.transpose(1, 2).view(B, self.decoder_dim, Hp, Wp)

        # Cross-attention fusion.
        feat_geo, feat_seg = self.cross_attention(feat_geo, feat_seg)

        # Predictions.
        out_dict["geo_map"] = self.predict_from_interactive_features(feat_geo, hs_geo, self.hyper_mlps_geo, (H, W))
        seg_raw = self.predict_from_interactive_features(feat_seg, hs_seg, self.hyper_mlps_seg, (H, W))
        out_dict["nuclei_binary_map"] = seg_raw[:, 0:1]

        return out_dict
