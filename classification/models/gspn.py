import os

import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

try:
    from .modules import normalize_w, GRN, LayerNorm2d, Linear2d, Mlp, gMlp, Linear2d, LayerNorm2d, PatchMerging2D, Permute
except:
    from modules import normalize_w, GRN, LayerNorm2d, Linear2d, Mlp, gMlp, Linear2d, LayerNorm2d, PatchMerging2D, Permute

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(os.path.join(project_root, 'ops')) # Or directly modify THIS
from ops.gaterecurrent.gaterecurrent2dnoind import GateRecurrent2dnoind

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class GSPNv1:
    def __initgspnv1__(
        self,
        # basic dims ===========
        feat_size,
        items_each_chunk=8,
        d_model=96,
        ssm_ratio=2.0,
        ssm_d_state=16,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # ======================
        n_directions=4,
        # ======================
        channel_first=True,
        is_glayers=False,
        force_fp32=True,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        self.d_state = ssm_d_state
        self.channel_first = channel_first

        self.c_group = 12
        self.n_directions = n_directions

        d_inner = int(ssm_ratio * d_model)

        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        self.forward = self.forwardv1
        self.forward_core = self.forward_corev1

        if is_glayers:
            self.items_each_chunk = feat_size
        else:
            self.items_each_chunk = items_each_chunk

        # in proj =======================================
        d_proj = d_inner
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        self.d_spn = d_inner
            
        self.conv2d = nn.Conv2d(
            in_channels=self.d_spn,
            out_channels=self.d_spn,
            groups=self.d_spn,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # spn module ============================
        self.spn_core = GateRecurrent2dnoind(self.items_each_chunk)

        # w, l, u proj ============================
        ks = 1
        self.x_conv_down = nn.Conv2d(self.d_spn, self.d_state, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.w_conv_up = nn.Conv2d(self.d_state, self.c_group * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.l_conv_up = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.u_conv_up = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.d_conv = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.m_conv = nn.Conv2d(self.n_directions, 1, kernel_size=1, bias=False)
        # out proj =======================================
        self.grn = GRN(d_inner)
        self.out_act = nn.Identity()
        self.out_norm = LayerNorm(self.d_spn)
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def spn_block(self, X, l, u, Gl, Gm, Gr, D=None, spn_module=None):
        Gl, Gm, Gr = normalize_w(Gl, Gm, Gr)
        
        Gl = Gl.to(X.dtype)
        Gm = Gm.to(X.dtype)
        Gr = Gr.to(X.dtype)
        
        out = spn_module(X, l, Gl, Gm, Gr)
        if D is not None:
            out = out * u + X * D
        else:
            out = out * u
        return out

    def forward_corev1(self, x: torch.Tensor=None, **kwargs):
        B, D, H, W = x.shape
        
        x_proxy = self.x_conv_down(x)
        ws = self.w_conv_up(x_proxy)
        Ls = self.l_conv_up(x_proxy).contiguous()
        Us = self.u_conv_up(x_proxy).contiguous()
        Ds = self.d_conv(x_proxy).contiguous()

        x_hwwh = torch.stack([x, x.transpose(2, 3).contiguous()], dim=1) 
        xs = torch.cat([x_hwwh, x_hwwh.flip(dims=[-1]).contiguous()], dim=1) # (b, k, d, h, w)
        xs = xs.view(B, -1, H, W) # (b, k, d, h, w)
        xs = xs.contiguous()

        Gs = torch.split(ws, D*self.n_directions, dim=1) # 3 * (b, d, h, w)
        G3 = [g.contiguous() for g in Gs]

        out_y = self.spn_block(xs, Ls, Us, G3[0], G3[1], G3[2], Ds, self.spn_core)

        out_y = out_y.view(B, self.n_directions, D*H, W)
        out_y = self.m_conv(out_y).view(B, D, H, W)

        y = self.out_norm(out_y)

        return y

    def forwardv1(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        x = self.conv2d(x) # (b, d, h, w)
        y = self.forward_core(x)
        y = self.out_act(y)
        y = self.grn(y)
        out = self.dropout(self.out_proj(y))
        return out

class SS2D(nn.Module, GSPNv1):
    def __init__(
        self,
        # basic dims ===========
        feat_size,
        items_each_chunk=8,
        d_model=96,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v1",
        channel_first=False,
        is_glayers=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            feat_size=feat_size, items_each_chunk=items_each_chunk, d_model=d_model, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, is_glayers=is_glayers,
        )
        self.__initgspnv1__(**kwargs) # gspn

# =====================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        feat_size,
        items_each_chunk=8,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_ratio=2.0,
        ssm_d_state=16,
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v1",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        is_glayers: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.is_lastlayers = False

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                feat_size=feat_size, 
                items_each_chunk=items_each_chunk,
                d_model=hidden_dim, 
                ssm_ratio=ssm_ratio,
                ssm_d_state=ssm_d_state,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
                is_glayers=is_glayers,
            )
        
        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        # FFN
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self, 
        feat_sizes,
        items_each_chunk=8,
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v1",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        specific_glayers=[2,3],
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.feat_sizes = feat_sizes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            elu=nn.ELU,
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                feat_size=self.feat_sizes[i_layer],
                items_each_chunk=items_each_chunk,
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                is_glayers=(i_layer in specific_glayers),
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        feat_size,
        items_each_chunk=8,
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,     
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v1",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        is_glayers=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                feat_size=feat_size,
                items_each_chunk=items_each_chunk,
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                is_glayers=(is_glayers and (d < 7 or (d+1)%7==0)),
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1).contiguous() if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):
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
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)