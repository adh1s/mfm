import os
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from mfm.SI.samplers import kernel_sampler_fn, t0_sampler_fn

torch.set_float32_matmul_precision("high")
import math
import time

import numpy as np
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from mfm.data import get_data_module
from mfm.models.model_wrapper import SIModelWrapper
from mfm.utils.evaluation import posterior_sampling_fn
from mfm.utils.steering import get_imagenet_vae_fn


@hydra.main(
    config_path="../conf/", config_name="config_sample.yaml", version_base="1.3"
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    ckpt_path = cfg.checkpoint_path

    name = f"uncond_{cfg.kernel_sampler_steps}_steps_cfg_{cfg.cfg_scale}"

    t = time.strftime("%Y%m%d-%H%M%S")
    name = f"{name}_{t}"
    log_dir = Path(ckpt_path).parent / f"samples_{name}"
    print(f"Logging to {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    sample_folder_dir = log_dir / "samples"
    os.makedirs(sample_folder_dir, exist_ok=True)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = cfg.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # instantiate model
    model = instantiate(cfg.model)
    SI = instantiate(cfg.SI)
    model = SIModelWrapper(model, SI, cfg.use_parametrization)
    model.to(device)

    # load checkpoint
    map_location = {"cuda:%d" % 0: "cuda:%d" % device}
    
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    checkpoint = {k[6:] if k.startswith("model.") else k: v for k, v in checkpoint.items()}
    missing, _ = model.load_state_dict(checkpoint, strict=False)
    print(f"Rank {rank}: Loaded checkpoint with missing keys: {missing}")

    model.eval()
    print(f"Rank {rank}: Loaded checkpoint with missing keys: {missing}")

    n = cfg.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(
        math.ceil(cfg.num_samples / global_batch_size) * global_batch_size
    )

    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    latent_shape = (cfg.model.in_channels, cfg.model.input_size, cfg.model.input_size)
    image_shape = (3, cfg.dataset.img_resolution, cfg.dataset.img_resolution)
    _, decode_fn = get_imagenet_vae_fn(device)
    encoder_fn, decoder_fn = get_imagenet_vae_fn(device)

    if cfg.classes:  # filter
        classes = torch.tensor(cfg.classes, device=device)
        class_labels = classes[
            torch.randint(
                0, len(classes), (samples_needed_this_gpu,), device=device
            )
        ]
    else:  # all classes
        class_labels = torch.randint(
            0, 1000, (samples_needed_this_gpu,), device=device
        )

        kwargs = {"class_labels": class_labels, "cfg_scale": cfg.cfg_scale}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if cfg.sampler == "kernel":
            samples = kernel_sampler_fn(
                model,
                shape=latent_shape,
                shape_decoded=image_shape,
                SI=SI,
                n_samples=samples_needed_this_gpu,
                n_batch_size=cfg.per_proc_batch_size,
                n_steps=cfg.kernel_sampler_steps,
                inverse_scaler_fn=decode_fn,
                **kwargs,
            )
        elif cfg.sampler == "t0":
            samples = t0_sampler_fn(
                model,
                shape=latent_shape,
                shape_decoded=image_shape,
                SI=SI,
                n_samples=samples_needed_this_gpu,
                n_batch_size=cfg.per_proc_batch_size,
                n_steps=cfg.kernel_sampler_steps,
                inverse_scaler_fn=decode_fn,
                **kwargs,
            )
    
    samples = (
        samples.to("cpu")
        .mul(255.0)
        .clamp(0, 255)
        .permute(0, 2, 3, 1)
        .to(dtype=torch.uint8)
        .numpy()
    )

    if cfg.save_png:
        for i, sample in tqdm(
            enumerate(samples),
            desc=f"Rank {rank} Saving Samples",
            total=samples_needed_this_gpu,
        ):
            index = rank * samples_needed_this_gpu + i
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

    dist.barrier()
    samples_tensor = torch.as_tensor(
        samples, dtype=torch.uint8, device=device
    ).contiguous()
    sample_gather_list = [
        torch.empty_like(samples_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(sample_gather_list, samples_tensor)

    labels_tensor = torch.as_tensor(
        class_labels, dtype=torch.int64, device=device
    ).contiguous()
    label_gather_list = [
        torch.empty_like(labels_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(label_gather_list, labels_tensor)

    if rank == 0:
        gathered = torch.cat(sample_gather_list, dim=0)[: cfg.num_samples]
        npz_path = f"{sample_folder_dir}/samples.npz"
        np.savez(npz_path, arr_0=gathered.to("cpu", dtype=torch.uint8).numpy())

        gathered_labels = torch.cat(label_gather_list, dim=0)[: cfg.num_samples]
        labels_npz_path = f"{sample_folder_dir}/labels.npz"
        np.savez(
            labels_npz_path,
            arr_0=gathered_labels.to("cpu", dtype=torch.int64).numpy(),
        )

        print(f"OUTPUT_PATH:{npz_path}")


if __name__ == "__main__":
    main()
