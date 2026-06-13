import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of a 4D (B, C, H, W) tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU between hidden layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CrossAttentionModule(nn.Module):
    """Bidirectional cross-attention between two feature maps of shape (B, C, H, W).
    Each stream attends to the other; outputs keep the original spatial layout."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn_a_to_b = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_b_to_a = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        B, C, H, W = feat_a.shape
        q_a = feat_a.flatten(2).transpose(1, 2)
        q_b = feat_b.flatten(2).transpose(1, 2)
        out_b, _ = self.attn_a_to_b(q_b, q_a, q_a)
        q_b = self.norm_b(q_b + out_b)
        out_a, _ = self.attn_b_to_a(q_a, q_b, q_b)
        q_a = self.norm_a(q_a + out_a)
        return q_a.transpose(1, 2).reshape(B, C, H, W), q_b.transpose(1, 2).reshape(B, C, H, W)


class BaseCellSeg(nn.Module):
    """Minimal base for cell-segmentation models.

    Holds shared attributes (patch size, class counts) and is subclassed by the
    concrete model, which defines its own encoder, decoder and post-processing.
    Renamed from a third-party method's class name to avoid confusion in this release.
    Several constructor arguments are accepted for signature compatibility and are
    not used directly here.
    """

    def __init__(self, num_nuclei_classes, num_tissue_classes, embed_dim,
                 input_channels, depth, num_heads, extract_layers,
                 mlp_ratio, qkv_bias, drop_rate, regression_loss):
        super().__init__()
        self.patch_size = 16
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.embed_dim = embed_dim
        self.regression_loss = regression_loss
