# Modified from LinFusion https://github.com/Huage001/LinFusion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

from diffusers import AutoPipelineForText2Image
import torch
import gc

from src.fusion import GSPNFusion, LinFusion
from src.tools import (
    forward_unet_wrapper, 
    forward_resnet_wrapper, 
    forward_crossattndownblock2d_wrapper, 
    forward_crossattnupblock2d_wrapper,
    forward_downblock2d_wrapper, 
    forward_upblock2d_wrapper,
    forward_transformer_block_wrapper)

root_path = './samples/sd15'
prompt = "Two dogs curled up asleep on a couch."
seed = 234


save_path = os.path.join(root_path, prompt)

if not os.path.exists(save_path):
    os.mkdir(save_path)

device = torch.device('cuda')

torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()
pipeline = AutoPipelineForText2Image.from_pretrained(
    "Lykon/dreamshaper-8", torch_dtype=torch.float16, variant="fp16"
).to(device)
GSPNFusion = GSPNFusion.construct_for(pipeline)
generator = torch.manual_seed(seed)
pipeline.enable_vae_tiling()
start_time = time.perf_counter()
image = pipeline(prompt=prompt,
            height=1024, width=1024, device=device, 
            num_inference_steps=50, guidance_scale=7.5,
            cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, gaussian_sigma=0.8,
            generator=generator, upscale_strength=0.56).images[0]
end_time = time.perf_counter()
running_time = (end_time - start_time) / 60
peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
print(f"Peak memory: {peak_memory:.2f} GB, Running time: {running_time:.3f} min")
image.save(save_path+f'/output_1k_{seed}.jpg')


torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()
generator = torch.manual_seed(seed)
start_time = time.perf_counter()
image = pipeline(image=image, prompt=prompt,
            height=2048, width=2048, device=device, 
            num_inference_steps=50, guidance_scale=7.5,
            cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, gaussian_sigma=0.8,
            generator=generator, upscale_strength=0.48).images[0]
end_time = time.perf_counter()
running_time = (end_time - start_time) / 60
peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
print(f"Peak memory: {peak_memory:.2f} GB, Running time: {running_time:.3f} min")
image.save(save_path+f'/output_2k_{seed}.jpg')


for _, _module in pipeline.unet.named_modules():
    if _module.__class__.__name__ == 'BasicTransformerBlock':
        _module.set_chunk_feed_forward(16, 1)
        _module.forward = forward_transformer_block_wrapper(_module)
    elif _module.__class__.__name__ == 'ResnetBlock2D':
        _module.nonlinearity.inplace = True
        _module.forward = forward_resnet_wrapper(_module)
    elif _module.__class__.__name__ == 'CrossAttnDownBlock2D':
        _module.forward = forward_crossattndownblock2d_wrapper(_module)
    elif _module.__class__.__name__ == 'DownBlock2D':
        _module.forward = forward_downblock2d_wrapper(_module)
    elif _module.__class__.__name__ == 'CrossAttnUpBlock2D':
        _module.forward = forward_crossattnupblock2d_wrapper(_module)
    elif _module.__class__.__name__ == 'UpBlock2D':
        _module.forward = forward_upblock2d_wrapper(_module)   

pipeline.unet.forward = forward_unet_wrapper(pipeline.unet)

torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()
generator = torch.manual_seed(seed)
start_time = time.perf_counter()
image = pipeline(image=image, prompt=prompt,
            height=4096, width=4096, device=device, 
            num_inference_steps=50, guidance_scale=7.5,
            cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, gaussian_sigma=0.8,
            generator=generator, upscale_strength=0.40).images[0]
end_time = time.perf_counter()
running_time = (end_time - start_time) / 60
peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
print(f"Peak memory: {peak_memory:.2f} GB, Running time: {running_time:.3f} min")
image.save(save_path+f'/output_4k_{seed}.jpg')