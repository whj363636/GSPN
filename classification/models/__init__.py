import os
from functools import partial

import torch

from .gspn import VSSM


def build_gspn_model(config, is_pretrain=False):
    model = VSSM(
        feat_sizes=config.MODEL.GSPN.FEAT_SIZE,
        items_each_chunk=config.MODEL.GSPN.ITEMS_EACH_CHUNK,
        patch_size=config.MODEL.GSPN.PATCH_SIZE, 
        in_chans=config.MODEL.GSPN.IN_CHANS, 
        num_classes=config.MODEL.NUM_CLASSES, 
        depths=config.MODEL.GSPN.DEPTHS, 
        dims=config.MODEL.GSPN.EMBED_DIM, 
        # ===================
        ssm_d_state=config.MODEL.GSPN.D_STATE,
        ssm_ratio=config.MODEL.GSPN.RATIO,
        ssm_rank_ratio=config.MODEL.GSPN.RANK_RATIO,
        ssm_act_layer=config.MODEL.GSPN.ACT_LAYER,
        ssm_conv=config.MODEL.GSPN.CONV,
        ssm_conv_bias=config.MODEL.GSPN.CONV_BIAS,
        ssm_drop_rate=config.MODEL.GSPN.DROP_RATE,
        ssm_init=config.MODEL.GSPN.INIT,
        forward_type=config.MODEL.GSPN.FORWARDTYPE,
        # ===================
        mlp_ratio=config.MODEL.GSPN.MLP_RATIO,
        mlp_act_layer=config.MODEL.GSPN.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.GSPN.MLP_DROP_RATE,
        # ===================
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.GSPN.PATCH_NORM,
        norm_layer=config.MODEL.GSPN.NORM_LAYER,
        downsample_version=config.MODEL.GSPN.DOWNSAMPLE,
        patchembed_version=config.MODEL.GSPN.PATCHEMBED,
        gmlp=config.MODEL.GSPN.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        specific_glayers=config.MODEL.GSPN.SPECIFIC_GLAYERS,
        # ===================
        posembed=config.MODEL.GSPN.POSEMBED,
        imgsize=config.DATA.IMG_SIZE,
    )
    return model

    return None


def build_model(config, is_pretrain=False):
    model = build_gspn_model(config, is_pretrain)
    return model




