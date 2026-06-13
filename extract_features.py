import os
import glob
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from tqdm import tqdm
import math
import json

# =========================================================
# 1. Generic image dataset
# =========================================================
class GenericImageDataset(Dataset):
    def __init__(self, root_dir=None, file_list=None, img_size=512, ext_list=['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
        """
        Args:
            root_dir: image root (the dataset's processed directory).
            file_list: list of file names / IDs (from JSON).
            img_size: target input size.
            ext_list: supported file extensions.
        """
        self.img_size = img_size
        self.files = []
        self.ext_list = ext_list

        # Mode 1: explicit list of IDs.
        if file_list is not None:
            self.files = file_list
            self.root_base = root_dir if root_dir else ""
            print(f"Initialized from list with {len(self.files)} target entries.")

        # Mode 2: scan a directory for images.
        elif root_dir is not None:
            self.root_base = root_dir
            image_dir = os.path.join(root_dir, 'images')
            if os.path.exists(image_dir):
                print(f"Scanning {image_dir} for images...")
                for f in os.listdir(image_dir):
                    if os.path.splitext(f)[1].lower() in ext_list:
                        self.files.append(f)
            else:
                # Fallback: scan root_dir itself.
                print(f"Scanning {root_dir} for images...")
                for f in os.listdir(root_dir):
                    if os.path.splitext(f)[1].lower() in ext_list:
                        self.files.append(f)

            print(f"Found {len(self.files)} images.")

        else:
            raise ValueError("Either root_dir or file_list must be provided.")

        if len(self.files) == 0:
            raise ValueError("No images found to process.")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        entry = self.files[idx]  # usually an ID, e.g. "0a7d30..."

        full_path = None

        # Strip path and extension to get the bare ID.
        query_id = os.path.splitext(os.path.basename(entry))[0]

        # Primary rule: look for {id}.png under root_dir/images/.
        if self.root_base:
            base_dir = os.path.join(self.root_base, 'images')

            # Try .png first (most common).
            candidate = os.path.join(base_dir, f"{query_id}.png")
            if os.path.exists(candidate):
                full_path = candidate
            else:
                # Try the other extensions.
                for ext in self.ext_list:
                    if ext == '.png': continue
                    candidate = os.path.join(base_dir, f"{query_id}{ext}")
                    if os.path.exists(candidate):
                        full_path = candidate
                        break

        # Fallback: the file may sit directly under root_dir.
        if full_path is None and self.root_base:
            for ext in self.ext_list:
                candidate = os.path.join(self.root_base, f"{query_id}{ext}")
                if os.path.exists(candidate):
                    full_path = candidate
                    break

        # Still not found.
        if full_path is None:
            # Last resort: maybe entry is itself an absolute path.
            if os.path.isabs(entry) and os.path.exists(entry):
                full_path = entry
            else:
                print(f"Warning: Could not find image for ID: {query_id}. Expected at: {os.path.join(self.root_base, 'images', query_id + '.png')}")
                return torch.zeros((3, self.img_size, self.img_size)), query_id

        # File ID used for saving -> the bare query_id.
        file_id = query_id

        try:
            img = Image.open(full_path).convert('RGB')
            tensor = self.transform(img)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            tensor = torch.zeros((3, self.img_size, self.img_size))

        return tensor, file_id

# =========================================================
# 2. Model wrapper
# =========================================================
class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        print(f"Loading model: {model_name}...")
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True
        )
        self.model.eval()

        if hasattr(self.model, 'patch_embed'):
            self.patch_size = self.model.patch_embed.patch_size[0]
        else:
            self.patch_size = 14

        print(f"Model Patch Size: {self.patch_size}")

    def forward(self, x):
        features = self.model.forward_features(x)

        if isinstance(features, dict):
            features = features['x_norm_clstoken']

        B, L, C = features.shape
        H_in, W_in = x.shape[2], x.shape[3]

        h_feat = H_in // self.patch_size
        w_feat = W_in // self.patch_size

        num_spatial = h_feat * w_feat

        cls_token = features[:, 0, :]
        spatial_tokens = features[:, -num_spatial:, :]

        spatial_features = spatial_tokens.transpose(1, 2).reshape(B, C, h_feat, w_feat)

        return spatial_features, cls_token

# =========================================================
# 3. Extraction loop
# =========================================================
def process_loader(loader, model, device, output_dir, fp16):
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if fp16 else torch.no_grad()

    print(f"  -> Saving features to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    with torch.inference_mode():
        for images, file_ids in tqdm(loader, desc="Extracting"):
            images = images.to(device, non_blocking=True)

            with autocast_ctx:
                spatial_feats, cls_tokens = model(images)

            spatial_feats = spatial_feats.float().cpu()
            cls_tokens = cls_tokens.float().cpu()

            for i, file_id in enumerate(file_ids):
                # Save as {ID}.pt under the corresponding phase directory.
                save_path = os.path.join(output_dir, f"{file_id}.pt")

                torch.save({
                    'spatial': spatial_feats[i].clone(),
                    'cls': cls_tokens[i].clone()
                }, save_path)

# =========================================================
# 4. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Universal DINOv3 Feature Extractor")
    parser.add_argument('--input_dir', type=str, default='./datasets/cryonuseg/processed', help="Root directory containing images (Scan Mode)")
    parser.add_argument('--json_path', type=str, default='./datasets/cryonuseg/processed/ids.json', help="Path to split.json (JSON Mode)")
    parser.add_argument('--output_dir', type=str, default='./precomputed_feats/cryonuseg', help="Directory to save .pt files")
    parser.add_argument('--model_name', type=str, default='vit_7b_patch16_dinov3.lvd1689m', help="Timm model name")
    parser.add_argument('--img_size', type=int, default=512, help="Input image size")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fp16', action='store_true', help="Use Float16 inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model.
    model = DinoV3FeatureExtractor(args.model_name).to(device)

    # Mode 1: JSON config (train/valid/test).
    if args.json_path:
        print(f"Mode: JSON Config ({args.json_path})")
        with open(args.json_path, 'r') as f:
            splits = json.load(f)

        # Iterate over each key in the JSON (train, valid, test, etc.).
        for phase, file_list in splits.items():
            if not isinstance(file_list, list): continue

            print(f"\n=== Processing Phase: {phase} ({len(file_list)} files) ===")

            dataset = GenericImageDataset(
                root_dir=args.input_dir,  # base path used to locate images
                file_list=file_list,
                img_size=args.img_size
            )

            if len(dataset) == 0: continue

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

            # Auto-create a sub-directory per phase: output_dir/train, output_dir/test, ...
            phase_output_dir = os.path.join(args.output_dir, phase)
            process_loader(loader, model, device, phase_output_dir, args.fp16)

    # Mode 2: directory scan.
    else:
        print(f"Mode: Directory Scan ({args.input_dir})")
        dataset = GenericImageDataset(
            root_dir=args.input_dir,
            img_size=args.img_size
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        process_loader(loader, model, device, args.output_dir, args.fp16)

    print("\nAll tasks completed.")

if __name__ == '__main__':
    main()
