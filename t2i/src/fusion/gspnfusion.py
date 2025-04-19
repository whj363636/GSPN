from diffusers.models.attention_processor import Attention
from diffusers import ModelMixin, ConfigMixin
import functools

from .gspn import GSPNmodule


model_dict = {
    "runwayml/stable-diffusion-v1-5": "/root/projects/GSPN/t2i/ckpt/gspnfusion_sd15/last_partial/iter-510000", # Change to huggingface link
    "Lykon/dreamshaper-8": "/root/projects/GSPN/t2i/ckpt/gspnfusion_sd15/last_partial/iter-510000",
    "stabilityai/stable-diffusion-xl-base-1.0": "/root/projects/GSPN/t2i/ckpt/linfusion_sdxl/gspn/iter-770000",
}


def replace_submodule(model, module_name, new_submodule):
    path, attr = module_name.rsplit(".", 1)
    parent_module = functools.reduce(getattr, path.split("."), model)
    setattr(parent_module, attr, new_submodule)


class GSPNFusion(ModelMixin, ConfigMixin):
    def __init__(self, modules_list, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.modules_dict = {}
        self.register_to_config(modules_list=modules_list)

        depth = len(modules_list)
        for i, attention_config in enumerate(modules_list):
            dim_n = attention_config["dim_n"]
            try:
                pure_gspn = attention_config["pure_gspn"]
            except:
                pure_gspn = True
                
            gspn_block = GSPNmodule(
                query_dim=dim_n,
                out_dim=dim_n,
                items_each_chunk=4,
                scale_factor=8 if pure_gspn else 32,
                pure_gspn=pure_gspn,
                is_glayers=(i > depth - 3),
            )
            self.add_module(f"{i}", gspn_block)
            self.modules_dict[attention_config["module_name"]] = gspn_block

    @classmethod
    def get_default_config(
        cls,
        pipeline=None,
        unet=None,
    ):
        """
        Get the default configuration for the GSPNFusion model.
        """
        assert unet is not None or pipeline.unet is not None
        unet = unet or pipeline.unet
        modules_list = []
        for module_name, module in unet.named_modules():
            if not isinstance(module, Attention):
                continue
            if "attn1" not in module_name:
                continue
            dim_n = module.to_q.weight.shape[0]
            # modules_list.append((module_name, dim_n, module.heads))
            modules_list.append(
                {
                    "module_name": module_name,
                    "dim_n": dim_n,
                    "pure_gspn": True,
                }
            )
        return {"modules_list": modules_list}

    @classmethod
    def construct_for(
        cls,
        pipeline=None,
        unet=None,
        load_pretrained=True,
        pretrained_model_name_or_path=None,
        pipe_name_path=None,
    ) -> "GSPNFusion":
        """
        Construct a GSPNFusion object for the given pipeline.
        """
        assert unet is not None or pipeline.unet is not None
        unet = unet or pipeline.unet
        if load_pretrained:
            # Load from pretrained
            if not pretrained_model_name_or_path:
                pipe_name_path = pipe_name_path or pipeline._internal_dict._name_or_path
                pretrained_model_name_or_path = model_dict.get(pipe_name_path, None)
                if pretrained_model_name_or_path:
                    print(
                        f"Matching GSPNFusion '{pretrained_model_name_or_path}' for pipeline '{pipe_name_path}'."
                    )
                else:
                    raise Exception(
                        f"GSPNFusion not found for pipeline [{pipe_name_path}], please provide the path."
                    )
            GSPNfusion = (
                GSPNFusion.from_pretrained(pretrained_model_name_or_path)
                .to(unet.device)
                .to(unet.dtype)
            )
        else:
            # Create from scratch without pretrained parameters
            default_config = GSPNFusion.get_default_config(unet=unet)
            GSPNfusion = GSPNFusion(**default_config).to(unet.device).to(unet.dtype)
        GSPNfusion.mount_to(unet=unet)
        return GSPNfusion

    def mount_to(self, pipeline=None, unet=None) -> None:
        """
        Mounts the modules in the `modules_dict` to the given `pipeline`.
        """
        assert unet is not None or pipeline.unet is not None
        unet = unet or pipeline.unet
        for module_name, module in self.modules_dict.items():
            replace_submodule(unet, module_name, module)
        self.to(unet.device).to(unet.dtype)