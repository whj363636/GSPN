# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import einops
from functools import partial
from torch import Tensor
from typing import Optional
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import PatchEmbed
from fvcore.nn import flop_count
import copy

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(project_root, 'ops')) # Or directly modify THIS
from ops.gaterecurrent.gaterecurrent2dnoind import GateRecurrent2dnoind

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, condition=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        if condition: 
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            ) 

    def forward(self, x, c=None): 
        if c is not None: 
            c = self.adaLN_modulation(c).squeeze(1)
            shift, scale = c.chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
        else:
            x = self.norm_final(x)
            x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Block(nn.Module):
    def __init__(
        self, input_size, hidden_size, 
        items_each_chunk=2, norm_cls=nn.LayerNorm, drop_path=0., 
        is_glayers=False, 
        skip=False,
    ):
        super().__init__()

        if is_glayers:
            self.items_each_chunk = input_size
        else:
            self.items_each_chunk = items_each_chunk

        self.d_state = hidden_size // 16

        self.in_proj = Linear2d(hidden_size, hidden_size, bias=False)
        self.conv2d = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            groups=hidden_size,
            bias=True,
            kernel_size=7,
            padding=3,
        )
        self.out_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.out_dconv = nn.Conv2d(hidden_size, hidden_size, groups=hidden_size, kernel_size=3, padding=1, bias=False)
        self.out_act = nn.ReLU()
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.spn_core = GateRecurrent2dnoind(items_each_chunk)
        self.x_conv_down = nn.Conv2d(hidden_size, self.d_state, kernel_size=1, bias=False)
        self.w_conv_up = nn.Conv2d(self.d_state, 12 * hidden_size, kernel_size=1, bias=False)
        self.l_conv_up = nn.Conv2d(self.d_state, 4 * hidden_size, kernel_size=1, bias=False)
        self.u_conv_up = nn.Conv2d(self.d_state, 4 * hidden_size, kernel_size=1, bias=False)
        self.d_conv = nn.Conv2d(self.d_state, 4 * hidden_size, kernel_size=1, bias=False)
        self.m_conv = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        
        self.norm = norm_cls(hidden_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None

    def normalize_w(self, Gl, Gm, Gr):
        Gl_s = torch.sigmoid(Gl)
        Gm_s = torch.sigmoid(Gm)
        Gr_s = torch.sigmoid(Gr)

        sum_s = Gl_s + Gm_s + Gr_s

        sum_s[:, :, 0, :] = Gm_s[:, :, 0, :] + Gr_s[:, :, 0, :]
        sum_s[:, :, -1, :] = Gl_s[:, :, -1, :] + Gm_s[:, :, -1, :]

        sum_s = sum_s.clamp(min=1e-7)

        return Gl_s / sum_s, Gm_s / sum_s, Gr_s / sum_s

    def spn_block(self, X, l, u, Gl, Gm, Gr, D=None, spn_module=None):
        X = X.to(l.dtype)
        Gl, Gm, Gr = self.normalize_w(Gl, Gm, Gr)
        out = spn_module(X, l, Gl, Gm, Gr)
        out = out * u

        if D is not None:
            out = out + X * D
        return out
    
    def spn(self, x):
        B, T, D = x.shape
        H = W = int(T ** 0.5)
        x = x.transpose(2, 1).contiguous().view(B, D, H, W) # (b, d, h, w)

        x = self.in_proj(x)
        x = self.conv2d(x) # (b, d, h, w)

        x_proxy = self.x_conv_down(x)
        ws = self.w_conv_up(x_proxy)
        Ls = self.l_conv_up(x_proxy).contiguous()
        Us = self.u_conv_up(x_proxy).contiguous()
        Ds = self.d_conv(x_proxy).contiguous()

        x_hwwh = torch.stack([x, x.transpose(2, 3).contiguous()], dim=1) 
        xs = torch.cat([x_hwwh, x_hwwh.flip(dims=[-1]).contiguous()], dim=1) # (b, k, d, h, w)
        xs = xs.view(B, -1, H, W) # (b, k, d, h, w)
        xs = xs.contiguous()

        Gs = torch.split(ws, D*4, dim=1) # 3 * (b, d, h, w)
        G3 = [g.contiguous() for g in Gs]

        out_y = self.spn_block(xs, Ls, Us, G3[0], G3[1], G3[2], Ds, self.spn_core)
        out_y = out_y.view(B, 4, D*H, W)
        out_y = self.m_conv(out_y)

        out_y = out_y.view(B, D, H, W)
        return out_y

    def forward(self, hidden_states, shortcut=None, skip=None):
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))

        if shortcut is None:
            shortcut = hidden_states
        else:
            shortcut = shortcut + self.drop_path(hidden_states)
        
        hidden_states = self.norm(shortcut.to(dtype=self.norm.weight.dtype))
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # out_states = self.spn(modulate(hidden_states, shift, scale))
        out_states = self.spn(hidden_states)

        out_states = self.out_conv(out_states)
        out_states = self.out_dconv(out_states)
        out_states = out_states * self.out_act(out_states)
        B, D, H, W = out_states.shape
        out_states = out_states.view(B, D, H*W).transpose(2, 1).contiguous() # (b, t, d)
        out_states = self.out_proj(out_states)

        return out_states, shortcut


def create_block(
    img_size,
    hidden_size,
    norm_epsilon=1e-5,
    drop_path=0.,
    skip=False,
    layer_idx=None,
    is_glayers=False,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        img_size,
        hidden_size,
        norm_cls=norm_cls,
        drop_path=drop_path,
        is_glayers=is_glayers,
        skip=skip,
    )
    block.layer_idx = layer_idx
    return block


class GSPN(nn.Module): 
    def __init__(
        self,
        img_size=32,
        patch_size=16,
        hidden_size=192,
        channels=3,
        depth=12,
        num_classes=-1,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5, 
        device=None,
        dtype=None,
        skip=True,
        no_aln_0=False,
        class_dropout_prob=0.1,
        **kwargs
    ): 
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.channels = channels
        self.patch_dim = patch_size ** 2 * channels * 2
        self.out_channels = channels * 2
        self.no_aln_0 = no_aln_0
        
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 

        self.x_embedder = PatchEmbed(img_size, patch_size, channels, hidden_size, bias=True)
        num_patches = (img_size // patch_size) ** 2
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        if self.num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.in_blocks = nn.ModuleList([
            create_block(
                img_size,
                hidden_size,
                norm_epsilon=norm_epsilon,
                layer_idx=i,
                drop_path=inter_dpr[i],
                **kwargs,
            )
            for i in range(depth // 2)])

        self.mid_block = create_block(
                img_size,
                hidden_size,
                norm_epsilon=norm_epsilon,
                layer_idx=depth // 2,
                drop_path=inter_dpr[depth // 2],
                **kwargs,
            )

        self.out_blocks = nn.ModuleList([
            create_block(
                img_size,
                hidden_size,
                norm_epsilon=norm_epsilon,
                layer_idx=i + depth // 2 + 1,
                is_glayers=(i > depth // 2 - 3),
                drop_path=inter_dpr[i + depth // 2 + 1],
                skip=skip,
                **kwargs,
            )
            for i in range(depth // 2)])

        # output head
        self.norm_f = nn.LayerNorm(hidden_size, eps=norm_epsilon, **factory_kwargs)
        
        if num_classes > 0:
            if self.no_aln_0:
                self.final_layer = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
            else:
                self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, condition=True)
        else:
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels) 

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize x_embedder like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        if not self.no_aln_0:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y=None, low_mem=False): 
        x = self.x_embedder(x)
        t = self.t_embedder(t)

        if y is not None:
            y = self.y_embedder(y, self.training)
            c = t + y
        else:
            c = t

        c = c.unsqueeze(dim=1)

        shortcut = None
        hidden_states = x + c
        skips = []

        for blk in self.in_blocks: 
            if low_mem:
                hidden_states, shortcut = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(blk), hidden_states, shortcut)
            else:
                hidden_states, shortcut = blk(hidden_states, shortcut)
            skips.append(hidden_states)

        hidden_states, shortcut = self.mid_block(hidden_states, shortcut)

        for blk in self.out_blocks:
            if low_mem:
                hidden_states, shortcut = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(blk), hidden_states, shortcut, skips.pop())
            else:
                hidden_states, shortcut = blk(hidden_states, shortcut, skips.pop())

        if shortcut is None:
            shortcut = hidden_states
        else:
            shortcut = shortcut + self.drop_path(hidden_states)
        hidden_states = self.norm_f(shortcut.to(dtype=self.norm_f.weight.dtype))

        x = hidden_states
        if y is not None:
            if self.no_aln_0:
                x = self.final_layer(x+c)
            else:
                x = self.final_layer(x, c)
        else:
            x = self.final_layer(x, c)

        x = unpatchify(x, self.out_channels)
        return x


    def forward_with_cfg(self, x, t, y, cfg_scale=1.5):
        """
        Forward pass of GSPN, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


    def flops(self, shape=(3, 224, 224), verbose=False):
        def gspn_flop_jit(inputs, outputs, verbose=True):
            def flops_gspn_fn(B, D, H, W):
                flops = B * (H - 1) * (7 * W - 4) * D
                return flops
            if verbose:
                print_jit_input_names(inputs)
            B, D, H, W = inputs[0].type().sizes()
            N = inputs[2].type().sizes()[1]
            flops = flops_gspn_fn(B=B, D=D, H=H, W=W)
            return flops
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.GateRecurrent2dnoindFunction": partial(gspn_flop_jit, verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        y = torch.randint(1, 100, (1,)).to(next(model.parameters()).device)
        times_steps = torch.randint(1, 100, (1,)).to(next(model.parameters()).device)
        Gflops, unsupported = flop_count(model=model, inputs=(input, times_steps, y, False), supported_ops=supported_ops)

        del model, input, y, times_steps
        return sum(Gflops.values()) * 1e9

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   GSPN Configs                                  #
#################################################################################

def GSPN_S_2(**kwargs): 
    return GSPN(patch_size=2, hidden_size=368, depth=24, **kwargs) 

def GSPN_S_4(**kwargs): 
    return GSPN(patch_size=4, hidden_size=368, depth=24, **kwargs) 

def GSPN_B_2(**kwargs): 
    return GSPN(patch_size=2, hidden_size=900, depth=30, **kwargs) 

def GSPN_B_4(**kwargs): 
    return GSPN(patch_size=4, hidden_size=900, depth=30, **kwargs) 

def GSPN_L_2(**kwargs): 
    return GSPN(patch_size=2, hidden_size=1200, depth=56, **kwargs) 

def GSPN_L_4(**kwargs): 
    return GSPN(patch_size=4, hidden_size=1200, depth=56, **kwargs) 

def GSPN_XL_2(**kwargs): 
    return GSPN(patch_size=2, hidden_size=1500, depth=56, **kwargs) 

def GSPN_XL_4(**kwargs): 
    return GSPN(patch_size=4, hidden_size=1500, depth=56, **kwargs) 


GSPN_models = {
    'GSPN-XL/2': GSPN_XL_2,  'GSPN-XL/4': GSPN_XL_4,
    'GSPN-L/2':  GSPN_L_2,   'GSPN-L/4':  GSPN_L_4,
    'GSPN-B/2':  GSPN_B_2,   'GSPN-B/4':  GSPN_B_4,
    'GSPN-S/2':  GSPN_S_2,   'GSPN-S/4':  GSPN_S_4,
}