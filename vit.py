import math
import torch
from torch import nn
import torch.nn.functional as F


def compute_mask(H, W, window_size, shift_size, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_mask = torch.zeros((1, H, W, 1), device=device)
    cnt = 0
    for h in (slice(0, -window_size),
              slice(-window_size, -shift_size),
              slice(-shift_size, None)):
        for w in (slice(0, -window_size),
                  slice(-window_size, -shift_size),
                  slice(-shift_size, None)):
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, ws, ws, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')) \
        .masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B,
                     H // window_size,
                     W // window_size,
                     window_size,
                     window_size,
                     -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        ))
        coords_flat = coords.flatten(1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_index = relative_coords.sum(-1).view(-1)

        self.register_buffer('relative_position_index', relative_index)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        bias = self.relative_position_bias_table[self.relative_position_index]
        bias = bias.view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.size(0)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads,
                                    qkv_bias=True, dropout=attn_drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, mask=None):
        B, H, W, C = SwinTransformerBlock._shape(x)
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask)

        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    @staticmethod
    def _shape(x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        return B, H, W, C


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        return self.reduction(x)


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size, mlp_ratio, drop, attn_drop,
                 drop_path, downsample=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = window_size // 2

        Hp, Wp = input_resolution
        max_H = math.ceil(Hp / window_size) * window_size
        max_W = math.ceil(Wp / window_size) * window_size
        mask = compute_mask(max_H, max_W,
                            window_size, self.shift_size)
        self.register_buffer('attn_mask', mask)

        dpr = list(torch.linspace(0, drop_path, depth).numpy())
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, num_heads, window_size,
                shift_size=0 if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio, drop=drop,
                attn_drop=attn_drop, drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim) if downsample else None

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))
        _, C, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C)

        for blk in self.blocks:
            x = blk(x, self.attn_mask.to(x.device))

        x = x.view(B, Hp, Wp, C)[:, :H, :W, :].view(B, H * W, C)
        if self.downsample:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=None,
                 num_heads=None,
                 window_size=7, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1):
        super().__init__()
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.patches_resolution = (img_size // patch_size,
                                   img_size // patch_size)
        self.pos_drop = nn.Dropout(drop_rate)

        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = (self.patches_resolution[0] // (2 ** i),
                   self.patches_resolution[1] // (2 ** i))
            layer = BasicLayer(
                dim=embed_dim * 2 ** i,
                input_resolution=res,
                depth=depth,
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                downsample=PatchMerging if i < len(depths) - 1 else None
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim * 2 ** (len(depths) - 1))
        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_norm(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class SwinClassifier(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__()
        self.swin = SwinTransformer(**kwargs)
        self.norm = nn.LayerNorm(self.swin.num_features)
        self.head = nn.Linear(self.swin.num_features, num_classes)

    def forward(self, x):
        x = self.swin(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)
