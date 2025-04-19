import torch 
import logging
import argparse
import os 
import torchvision
import math 
import time
import random
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from torchvision import transforms
from glob import glob
from collections import OrderedDict
from copy import deepcopy
from PIL import Image
from tqdm import tqdm 
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL 

from gspn import GSPN_models 
from diffusion import create_diffusion


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args): 
    # Setup DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)

    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    print(args)

    # Variables for monitoring/logging purposes
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    
    # Setup an experiment folder
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    model_string_name = args.model.replace("/", "-")  # e.g., GSPN-XL/2 --> GSPN-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)

    if rank == 0:
        print(f"Experiment directory created at {experiment_dir}")

    # Setup data:
    features_dir = f"{args.feature_path}/imagenet{args.image_size}_features"
    labels_dir = f"{args.feature_path}/imagenet{args.image_size}_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if rank == 0:
        print(f"Dataset contains {len(dataset):,} images ({args.feature_path})")


    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    model = GSPN_models[args.model](
        img_size=latent_size,
        num_classes=args.num_classes,
        channels=4,
        no_aln_0=False,
    )
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    loss_scaler = NativeScalerWithGradNormCount()

    print(f'no checkpoint found in {checkpoint_dir}, ignoring auto resume')

    if args.ckpt:
        try:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            torch.cuda.empty_cache()
        except Exception as e:
            args.ckpt = second_to_last_checkpoint
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            torch.cuda.empty_cache()
        model.load_state_dict(checkpoint['model'], strict=True)
        ema.load_state_dict(checkpoint['ema'], strict=True)
        model = model.cuda()
        ema = ema.cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
        opt.load_state_dict(checkpoint['opt'])
        del checkpoint
        print(f"Using checkpoint: {args.ckpt}")
    else:
        model = model.cuda()
        ema = ema.cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    if rank == 0:
        print(f"GSPN Parameters: {sum(p.numel() for p in model.parameters()):,}")
        flops = model.flops(shape=(4, latent_size, latent_size))
        print(f"number of GFLOPs: {flops / 1e9}")
            
    if args.ckpt:
        train_steps = int(args.ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / (len(dataset) / args.global_batch_size)) + 1
        print(f"Initial state: step={train_steps}, epoch={start_epoch}")
    else:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights  

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

    model.train() 
    ema.eval()

    if rank == 0:
        print(f"Training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs): 
        sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"Beginning epoch {epoch}...")

        for data_iter_step, (x, y) in enumerate(loader): 

            adjust_learning_rate(opt, epoch, args)
            
            x = x.cuda(non_blocking=True)
            x = x.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device) 
            
            if args.num_classes > 0: 
                y = y.cuda(non_blocking=True)
                y = y.squeeze(dim=1)
            else:
                y = None 
            
            model_kwargs = dict(y=y) 
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            is_second_order = hasattr(opt, 'is_second_order') and opt.is_second_order
            grad_norm = loss_scaler(loss, opt, clip_grad=1.0,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step > 1))

            loss_scale_value = loss_scaler.state_dict()["scale"]
            torch.cuda.synchronize()
            
            opt.zero_grad()
            update_ema(ema, model.module)

            running_loss += loss.item() 
            loss_avg = running_loss / train_steps if train_steps > 0 else 0
            train_steps += 1
            log_steps += 1
            
            if rank == 0 and train_steps % args.log_every == 0:
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                print(
                    f'(step={train_steps:07d}/{len(loader)})\t'
                    f'Train Steps/Sec: {steps_per_sec:.2f}\t'
                    f'Loss {loss_avg:.4f}\t'
                    f'Mem {memory_used:.0f}MB')
                running_loss = 0
                log_steps = 0
                start_time = time.time()

        # if train_steps % args.ckpt_every == 0 and train_steps > 0:
        if (epoch + 1) % 5 == 0 or 'L' in args.model or 'XL' in args.model:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()



if __name__ == "__main__": 
    # pytorch major version (1.x or 2.x)
    PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(GSPN_models.keys()), default="GSPN-L/2")
    parser.add_argument("--global-seed", type=int, default=420) 
    
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=25000)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--min_lr', type=float, default=1e-6)

    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    else:
        parser.add_argument("--local_rank", type=int, default=int(os.environ['LOCAL_RANK']), help='local rank for DistributedDataParallel')
    
    args = parser.parse_args()
    main(args)
