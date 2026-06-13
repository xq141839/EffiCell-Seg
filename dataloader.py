import os
import random
import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt, maximum, find_objects
from skimage.segmentation import find_boundaries
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

# Dataset root (relative; override via constructor if needed).
DATA_ROOT = "./datasets"
VALID_PARADIGMS = ("tsfd", "hovernet", "stardist", "cellpose")


class UnifiedCellDataset(Dataset):
    def __init__(self, data_name, jsfiles, root_dir, phase='train',
                 mode='cached',  # 'cached' (load .pt features) or 'online' (load images)
                 filter_empty=False, paradigm='tsfd', n_rays=32,
                 img_size=512, data_root=DATA_ROOT):
        """
        Args:
            root_dir: in 'cached' mode points to the feature dir;
                      in 'online' mode points to the dataset root holding images.
            mode: 'cached' (load .pt features) or 'online' (run the backbone on images).
            paradigm: instance-seg target type, one of tsfd / hovernet / stardist.
            n_rays: number of radial directions for the stardist paradigm.
        """
        if paradigm not in VALID_PARADIGMS:
            raise ValueError(f"Unknown paradigm '{paradigm}'. Choose from {VALID_PARADIGMS}.")

        self.root_dir = root_dir
        self.phase = phase
        self.mode = mode
        self.paradigm = paradigm
        self.n_rays = n_rays
        self.img_size = img_size

        # Processed data layout: <data_root>/<data_name>/processed/{npy,images}
        self.processed_path = os.path.join(data_root, data_name, 'processed')

        if filter_empty:
            print(f"[{phase}] Filtering empty samples from {len(jsfiles)} files...")
            self.jsfiles = self._filter_empty_samples(jsfiles)
            print(f"[{phase}] Kept {len(self.jsfiles)} non-empty files.")
        else:
            self.jsfiles = jsfiles

        # Online-mode preprocessing (ImageNet statistics).
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _filter_empty_samples(self, jsfiles):
        valid_files = []
        for f in tqdm(jsfiles, desc="Checking empty masks"):
            image_id = f.split('.')[0]
            nuclei_path = os.path.join(self.processed_path, 'npy', image_id + '.npy')
            try:
                mask = np.load(nuclei_path)
                if mask.max() > 0:
                    valid_files.append(f)
            except Exception:
                pass
        return valid_files

    def __len__(self):
        return len(self.jsfiles)

    # === Geometric target generators ===

    def gen_dist_map(self, inst_map):
        """TSFD: normalized intra-instance distance map, shape (1, H, W).
        Vectorized over instances (no per-instance Python loop)."""
        if inst_map.max() == 0:
            return np.zeros((1, inst_map.shape[0], inst_map.shape[1]), dtype=np.float32)

        # Treat touching boundaries as background so the distance transform separates
        # adjacent instances.
        boundaries = find_boundaries(inst_map, mode='inner')
        binary_for_dist = (inst_map > 0) & (~boundaries)
        dist = distance_transform_edt(binary_for_dist)

        inst_ids = np.unique(inst_map)
        inst_ids = inst_ids[inst_ids != 0]
        if len(inst_ids) == 0:
            return dist[None, ...].astype(np.float32)

        # Per-instance max distance, then normalize each instance to [0, 1].
        max_vals = maximum(dist, labels=inst_map, index=inst_ids)
        max_map_lookup = np.zeros(inst_map.max() + 1, dtype=np.float32)
        max_map_lookup[inst_ids] = max_vals
        max_dist_map = max_map_lookup[inst_map]
        max_dist_map[max_dist_map == 0] = 1.0

        dist_norm = dist / max_dist_map
        return dist_norm[None, ...].astype(np.float32)

    def gen_hv_map(self, inst_map):
        """HoVer-Net: horizontal/vertical distance maps, shape (2, H, W), each in [-1, 1].
        Channel 0 = horizontal (x), channel 1 = vertical (y), measured from each
        instance centroid and normalized separately on the negative/positive sides."""
        H, W = inst_map.shape
        h_map = np.zeros((H, W), dtype=np.float32)
        v_map = np.zeros((H, W), dtype=np.float32)

        inst_ids = np.unique(inst_map)
        inst_ids = inst_ids[inst_ids != 0]
        if len(inst_ids) == 0:
            return np.stack([h_map, v_map]).astype(np.float32)

        inst_map_i = inst_map.astype(np.int32)
        slices = find_objects(inst_map_i)
        for inst_id in inst_ids:
            sl = slices[inst_id - 1]
            if sl is None:
                continue
            y0, x0 = sl[0].start, sl[1].start
            sub = inst_map_i[sl] == inst_id
            ys, xs = np.nonzero(sub)
            gy = ys + y0
            gx = xs + x0

            cy = gy.mean()
            cx = gx.mean()
            ry = (gy - cy).astype(np.float32)
            rx = (gx - cx).astype(np.float32)

            neg = rx < 0
            pos = rx > 0
            if neg.any():
                rx[neg] /= np.abs(rx[neg].min())
            if pos.any():
                rx[pos] /= rx[pos].max()
            neg = ry < 0
            pos = ry > 0
            if neg.any():
                ry[neg] /= np.abs(ry[neg].min())
            if pos.any():
                ry[pos] /= ry[pos].max()

            h_map[gy, gx] = rx
            v_map[gy, gx] = ry

        return np.stack([h_map, v_map]).astype(np.float32)

    def gen_star_dist_map(self, inst_map):
        """StarDist: radial boundary distances, shape (n_rays, H, W).
        For each foreground pixel, march along n_rays unit directions until leaving
        the instance; the step count is the distance. Ray convention matches the
        model decoder: phi_k = 2*pi*k/n_rays, direction = (sin phi, cos phi) in (row, col).

        NOTE: this is compute-heavy. For large datasets, precompute these targets once
        and cache them to disk rather than regenerating every epoch.
        """
        n_rays = self.n_rays
        H, W = inst_map.shape
        out = np.zeros((n_rays, H, W), dtype=np.float32)

        inst_ids = np.unique(inst_map)
        inst_ids = inst_ids[inst_ids != 0]
        if len(inst_ids) == 0:
            return out

        phis = 2.0 * np.pi * np.arange(n_rays) / n_rays
        sin_p = np.sin(phis).astype(np.float32)
        cos_p = np.cos(phis).astype(np.float32)

        inst_map_i = inst_map.astype(np.int32)
        slices = find_objects(inst_map_i)
        for inst_id in inst_ids:
            sl = slices[inst_id - 1]
            if sl is None:
                continue
            y0, x0 = sl[0].start, sl[1].start
            crop = inst_map_i[sl] == inst_id
            ch, cw = crop.shape
            ys, xs = np.nonzero(crop)  # foreground pixels in crop coords
            n_pix = ys.shape[0]
            max_steps = int(np.ceil(np.hypot(ch, cw))) + 1

            for k in range(n_rays):
                dy, dx = sin_p[k], cos_p[k]
                yy = ys.astype(np.float32).copy()
                xx = xs.astype(np.float32).copy()
                dist_k = np.zeros(n_pix, dtype=np.float32)
                done = np.zeros(n_pix, dtype=bool)
                for step in range(1, max_steps + 1):
                    yy += dy
                    xx += dx
                    yi = np.rint(yy).astype(np.int32)
                    xi = np.rint(xx).astype(np.int32)
                    inside = (yi >= 0) & (yi < ch) & (xi >= 0) & (xi < cw)
                    member = np.zeros(n_pix, dtype=bool)
                    member[inside] = crop[yi[inside], xi[inside]]
                    newly_out = (~member) & (~done)
                    dist_k[newly_out] = step
                    done |= newly_out
                    if done.all():
                        break
                dist_k[~done] = max_steps
                out[k, ys + y0, xs + x0] = dist_k

        return out

    def gen_flow_map(self, inst_map):
        """Cellpose: 2-channel flow field (dy, dx), shape (2, H, W). Flows are unit
        vectors pointing toward each instance's center, taken from the gradient of a
        within-instance heat diffusion (zero-flux outside the mask). Background is 0.

        NOTE: compute-heavy (iterative diffusion per instance). For large datasets,
        precompute these targets once and cache them to disk.
        """
        H, W = inst_map.shape
        mu = np.zeros((2, H, W), dtype=np.float32)  # (dy, dx)

        inst_ids = np.unique(inst_map)
        inst_ids = inst_ids[inst_ids != 0]
        if len(inst_ids) == 0:
            return mu

        inst_map_i = inst_map.astype(np.int32)
        slices = find_objects(inst_map_i)
        for inst_id in inst_ids:
            sl = slices[inst_id - 1]
            if sl is None:
                continue
            y0, x0 = sl[0].start, sl[1].start
            sub = inst_map_i[sl] == inst_id
            ly, lx = np.nonzero(sub)
            if ly.size == 0:
                continue

            # Local padded grid: a zero border acts as a Dirichlet boundary so heat
            # cannot leak past the instance.
            h_l, w_l = sub.shape
            T = np.zeros((h_l + 2, w_l + 2), dtype=np.float64)
            ly_p, lx_p = ly + 1, lx + 1

            # Center: the in-mask pixel closest to the median coordinate.
            y_med, x_med = np.median(ly), np.median(lx)
            c = np.argmin((ly - y_med) ** 2 + (lx - x_med) ** 2)
            cy, cx = ly_p[c], lx_p[c]

            n_iter = 2 * int((ly.max() - ly.min()) + (lx.max() - lx.min()) + 2)
            for _ in range(n_iter):
                T[cy, cx] += 1.0
                # Synchronous (Jacobi) 9-point diffusion over the instance pixels only.
                T[ly_p, lx_p] = (1.0 / 9.0) * (
                    T[ly_p, lx_p]
                    + T[ly_p - 1, lx_p] + T[ly_p + 1, lx_p]
                    + T[ly_p, lx_p - 1] + T[ly_p, lx_p + 1]
                    + T[ly_p - 1, lx_p - 1] + T[ly_p - 1, lx_p + 1]
                    + T[ly_p + 1, lx_p - 1] + T[ly_p + 1, lx_p + 1]
                )

            T = np.log(1.0 + T)
            dy = T[ly_p + 1, lx_p] - T[ly_p - 1, lx_p]
            dx = T[ly_p, lx_p + 1] - T[ly_p, lx_p - 1]
            norm = np.sqrt(dy ** 2 + dx ** 2) + 1e-20
            mu[0, ly + y0, lx + x0] = (dy / norm).astype(np.float32)
            mu[1, ly + y0, lx + x0] = (dx / norm).astype(np.float32)

        return mu

    def gen_geo_target(self, inst_map):
        """Dispatch to the active paradigm's geometric-target generator."""
        if self.paradigm == 'hovernet':
            return self.gen_hv_map(inst_map)
        if self.paradigm == 'stardist':
            return self.gen_star_dist_map(inst_map)
        if self.paradigm == 'cellpose':
            return self.gen_flow_map(inst_map)
        return self.gen_dist_map(inst_map)

    def apply_augmentations(self, input_tensor, inst_map, phase):
        """Jointly flip/rotate the input (features or image) and the instance map with
        the same random ops. Geometric and binary targets are derived AFTER this, so we
        never need to transform paradigm-specific maps (whose values are orientation
        dependent) directly."""
        if phase != 'train':
            return input_tensor, inst_map

        # input_tensor is (C, H, W); inst_map is (H, W). Matched axes:
        # horizontal -> tensor dim 2 / array axis 1; vertical -> dim 1 / axis 0.
        if random.random() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])
            inst_map = np.flip(inst_map, axis=1)
        if random.random() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[1])
            inst_map = np.flip(inst_map, axis=0)

        # rot90 over the spatial plane; torch dims [1, 2] match numpy axes (0, 1).
        k = random.randint(0, 3)
        if k > 0:
            input_tensor = torch.rot90(input_tensor, k, [1, 2])
            inst_map = np.rot90(inst_map, k)

        return input_tensor.contiguous(), np.ascontiguousarray(inst_map)

    def __getitem__(self, idx):
        image_id = self.jsfiles[idx].split('.')[0]

        input_data = {}
        if self.mode == 'cached':
            feat_path = os.path.join(self.root_dir, f"{image_id}.pt")
            data = torch.load(feat_path, map_location='cpu', weights_only=True)
            input_tensor = data['spatial']
            input_data['cls_token'] = data['cls']
        else:
            img_path = os.path.join(self.processed_path, 'images', f"{image_id}.png")
            if not os.path.exists(img_path):
                img_path = img_path.replace('.png', '.jpg')
            if not os.path.exists(img_path):
                img_path = img_path.replace('.jpg', '.tif')

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            input_tensor = self.transform(img)  # (3, H, W)

        nuclei_path = os.path.join(self.processed_path, 'npy', image_id)
        inst_map = np.load(nuclei_path + '.npy')
        if inst_map.ndim == 3:
            inst_map = inst_map[:, :, 0]
        if inst_map.shape != (self.img_size, self.img_size):
            inst_map = cv2.resize(inst_map, (self.img_size, self.img_size),
                                  interpolation=cv2.INTER_NEAREST)
        inst_map = inst_map.astype(np.int32)

        # Augment first, then derive binary + geometric targets from the augmented map.
        input_tensor, inst_map = self.apply_augmentations(input_tensor, inst_map, self.phase)

        binary_map = (inst_map > 0).astype(np.float32)
        geo_map = self.gen_geo_target(inst_map)

        geo_t = torch.from_numpy(geo_map).float()
        bin_t = torch.from_numpy(binary_map).float()

        if self.mode == 'cached':
            input_data['spatial_features'] = input_tensor
        else:
            input_data['image'] = input_tensor

        return {
            "image_id": image_id,
            **input_data,
            "geo_map": geo_t,
            "binary_map": bin_t,
            "inst_map": torch.from_numpy(inst_map).long(),
        }
