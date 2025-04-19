import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(os.path.join(project_root, 'ops')) # Or directly modify THIS
from ops.gaterecurrent.gaterecurrent2dnoind import GateRecurrent2dnoind


def get_none_linear_projection(query_dim, mid_dim=None):
    # If mid_dim is None, then the mid_dim is the same as query_dim
    # If mid_dim is -1, then no non-linear projection is used, and the identity is returned
    return (
        torch.nn.Sequential(
            torch.nn.Linear(query_dim, mid_dim or query_dim),
            torch.nn.LayerNorm(mid_dim or query_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(mid_dim or query_dim, query_dim),
        )
        if mid_dim != -1
        else torch.nn.Identity()
    )


class GSPNmodule(Attention):
    def __init__(self, *args, items_each_chunk=2, scale_factor=16, pure_gspn=True, is_glayers=None, **kwargs):
        """
        Args:
            query_dim: the dimension of the query.
            out_dim: the dimension of the output.
            dim_head: the dimension of the head. (dim_head * num_heads = query_dim)
            projection_mid_dim: the dimension of the intermediate layer in the non-linear projection. 
              If `None`, then the dimension is the same as the query dimension.
              If `-1`, then no non-linear projection is used, and the identity is returned.
        """
        super().__init__(*args, **kwargs)
        # self.query_dim = kwargs.get("query_dim")
        self.d_state = self.query_dim // scale_factor
        self.pure_gspn = pure_gspn

        if self.pure_gspn:
            # Unregister parameters to avoid the error
            if hasattr(self, 'to_q'):
                del self.to_q
            if hasattr(self, 'to_k'):
                del self.to_k
            if hasattr(self, 'to_v'):
                del self.to_v

        self.items_each_chunk = items_each_chunk

        self.spn_core = GateRecurrent2dnoind(items_each_chunk)
        # self.spn_core = GateRecurrent2d(items_each_chunk)

        self.x_conv_down = nn.Conv2d(self.query_dim, self.d_state, kernel_size=1, bias=False)
        self.w_conv_up = nn.Conv2d(self.d_state, 12 * self.query_dim, kernel_size=1, bias=False)
        self.l_conv_up = nn.Conv2d(self.d_state, 4 * self.query_dim, kernel_size=1, bias=False)
        self.u_conv_up = nn.Conv2d(self.d_state, 4 * self.query_dim, kernel_size=1, bias=False)
        self.d_conv = nn.Conv2d(self.d_state, 4 * self.query_dim, kernel_size=1, bias=False)
        self.m_conv = nn.Conv2d(4, 1, kernel_size=1, bias=False)

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
    
    def spn(self, x, Q=None, K=None, V=None):
        B, T, D = x.shape
        H = W = int(T ** 0.5)
        is_square = (H * W == T)

        if not is_square:
            W = int((T * 2) ** 0.5)
            H = W // 2

        x = x.transpose(2, 1).contiguous().view(B, D, H, W) # (b, d, h, w)
        x_proxy = self.x_conv_down(x)
        ws = self.w_conv_up(x_proxy)
        Ds = self.d_conv(x_proxy).contiguous()

        if self.pure_gspn:
            Ls = self.l_conv_up(x_proxy).contiguous()
            Us = self.u_conv_up(x_proxy).contiguous()
        else:
            Q = Q.transpose(2, 1).contiguous().view(B, D, H, W) # (b, d, h, w)
            K = K.transpose(2, 1).contiguous().view(B, D, H, W) # (b, d, h, w)
            V = V.transpose(2, 1).contiguous().view(B, D, H, W) # (b, d, h, w)

            l_proxy = self.x_conv_down(x+K)
            u_proxy = self.x_conv_down(x+Q)

            Ls = self.l_conv_up(l_proxy).contiguous()
            Us = self.u_conv_up(u_proxy).contiguous()

        x = x + V if not self.pure_gspn else x
        
        if is_square:
            x_hwwh = torch.stack([x, x.transpose(2, 3).contiguous()], dim=1) 
            xs = torch.cat([x_hwwh, x_hwwh.flip(dims=[-1]).contiguous()], dim=1) # (b, k, d, h, w)

            xs = xs.view(B, -1, H, W) # (b, k, d, h, w)
            xs = xs.contiguous()

            Gs = torch.split(ws, D*4, dim=1) # 3 * (b, d, h, w)
            G3 = [g.contiguous() for g in Gs]

            out_y = self.spn_block(xs, Ls, Us, G3[0], G3[1], G3[2], Ds, self.spn_core)

        else:
            x_hwwh = torch.stack([x, x.flip(dims=[-1])], dim=1) 
            x_hwwh_t = x_hwwh.transpose(2, 3).contiguous()

            x_hwwh = x_hwwh.view(B, -1, H, W) # (b, k, d, h, w)
            x_hwwh = x_hwwh.contiguous()

            x_hwwh_t = x_hwwh_t.view(B, -1, H, W) # (b, k, d, h, w)
            x_hwwh_t = x_hwwh_t.contiguous()

            Gs = torch.split(ws, D*2, dim=1) # 6 * (b, d, h, w)
            G6 = [g.contiguous() for g in Gs]

            out = self.spn_block(x_hwwh, Ls[:, :2*D, :, :].contiguous(), Us[:, :2*D, :, :].contiguous(), G6[0], G6[1], G6[2], Ds[:, :2*D, :, :], self.spn_core)
            out_f = self.spn_block(x_hwwh_t, Ls[:, 2*D:, :, :].contiguous(), Us[:, 2*D:, :, :].contiguous(), G6[3], G6[4], G6[5], Ds[:, 2*D:, :, :], self.spn_core)

            out = out.view(B, 2, D, H*W)
            out_f = out_f.view(B, 2, D, H*W)
            out_y = torch.cat([out, out_f], dim=1)

        out_y = out_y.view(B, 4, D*H, W)
        out_y = self.m_conv(out_y).view(B, D, T)
        out_y = out_y.transpose(2, 1).contiguous() # (b, t, d)

        return out_y

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if self.pure_gspn:
            hidden_states = self.spn(hidden_states)
        else:
            _, sequence_length, _ = hidden_states.shape

            query = self.to_q(hidden_states)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            hidden_states = self.spn(hidden_states, query, key, value)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states
