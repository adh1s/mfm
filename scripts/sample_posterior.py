import os
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision("high")

import math
import time

import numpy as np
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from mfm.data import get_data_module
from mfm.models.model_wrapper import SIModelWrapper
from mfm.utils.evaluation import posterior_sampling_fn
from mfm.utils.steering import get_imagenet_vae_fn


@hydra.main(
    config_path="../conf/",
    config_name="config_sample_posterior.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    ckpt_path = cfg.checkpoint_path
    time_string = time.strftime("%Y%m%d-%H%M%S")
    name = f"steps_{cfg.posterior_sampler_steps}_t_{str(cfg.posterior_t).replace('.', '_')}_{time_string}"

    log_dir = Path(ckpt_path).parent / f"posterior_samples_{name}"
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
    checkpoint = {
        k[6:] if k.startswith("model.") else k: v for k, v in checkpoint.items()
    }
    missing, _ = model.load_state_dict(checkpoint, strict=False)
    print(f"Rank {rank}: Loaded checkpoint with missing keys: {missing}")
    model.eval()

    # load in vae
    encode_fn, decode_fn = get_imagenet_vae_fn(device)

    n = cfg.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(
        math.ceil(cfg.num_samples / global_batch_size) * global_batch_size
    )

    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    image_shape = (3, cfg.dataset.img_resolution, cfg.dataset.img_resolution)

    # get data module!
    cfg.trainer.batch_size = cfg.per_proc_batch_size // cfg.posterior_n_samples

    print("Adjusted per-process batch size:", cfg.trainer.batch_size)
    datamodule = get_data_module(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    test_dataloader = datamodule.train_dataloader()
    n_classes = len(cfg.to_generate_classes) if cfg.to_generate_classes else 1

    orig_data = torch.zeros((samples_needed_this_gpu, *image_shape), device=device)
    xt = torch.zeros((samples_needed_this_gpu, *image_shape), device=device)

    samples = torch.zeros(
        (samples_needed_this_gpu, n_classes, cfg.posterior_n_samples, *image_shape),
        device=device,
    )
    class_labels = torch.zeros(
        (samples_needed_this_gpu, n_classes), device=device, dtype=torch.int64
    )

    n_samples = 0

    for _, batch in tqdm(
        enumerate(test_dataloader), desc=f"Rank {rank} Sampling Batches"
    ):
        start = n_samples
        curr_bs = min(samples_needed_this_gpu - start, cfg.trainer.batch_size)
        x, y = batch
        x = x.to(device=device)[:curr_bs]
        y = y.to(device=device)[:curr_bs]

        if len(cfg.filter_classes) > 0:
            mask = torch.zeros_like(y, dtype=torch.bool)
            for cls in cfg.filter_classes:
                mask = mask | (y == cls)
            x = x[mask]
            y = y[mask]
            samps = x.shape[0]

            if samps == 0:
                if (start + curr_bs) >= samples_needed_this_gpu:
                    break
                else:
                    continue

            curr_bs = min(curr_bs, samps)

        n_samples += curr_bs
        orig_data[start : start + curr_bs] = (x + 1) / 2.0

        # encode
        x = encode_fn(x)

        # sample noisy latent
        noise = torch.randn_like(x)
        xt_batch = noise * (1 - cfg.posterior_t) + x * cfg.posterior_t
        t_cond_batch = torch.full((x.shape[0],), cfg.posterior_t, device=device)

        # repeat for posterior sampling
        xt_batch = xt_batch.repeat_interleave(cfg.posterior_n_samples, dim=0)
        t_cond_batch = t_cond_batch.repeat_interleave(cfg.posterior_n_samples, dim=0)

        # generate posterior samples for all classes of interest
        ys_to_generate = []
        if cfg.to_generate_classes:
            for cls in cfg.to_generate_classes:
                ys_to_generate.append(
                    torch.full((curr_bs,), cls, device=device, dtype=y.dtype)
                )
        else:
            ys_to_generate.append(y)

        for cls_idx, y in enumerate(ys_to_generate):
            labels_batch = y.to(device=device)
            cfg_batch = torch.full((x.shape[0],), cfg.cfg_scale, device=device)

            labels_batch = labels_batch.repeat_interleave(
                cfg.posterior_n_samples, dim=0
            )
            cfg_batch = cfg_batch.repeat_interleave(cfg.posterior_n_samples, dim=0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    xt_out, x1_out = posterior_sampling_fn(
                        model,
                        xt_batch,
                        t_cond_batch,
                        n_samples_per_image=cfg.posterior_n_samples,
                        inverse_scaler=decode_fn,
                        labels=labels_batch,
                        cfg_scales=cfg_batch,
                        posterior_sampler="consistency",
                        n_steps=cfg.posterior_sampler_steps,
                    )

            # save
            samples[start : start + curr_bs, cls_idx] = x1_out
            class_labels[start : start + curr_bs, cls_idx] = y

            xt_out = xt_out.view(curr_bs, cfg.posterior_n_samples, *xt_out.shape[1:])
            xt[start : start + curr_bs] = xt_out[:, 0, ...]

        if n_samples >= samples_needed_this_gpu:
            break

    samples = torch.clamp(255.0 * samples, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    orig_data = (
        torch.clamp(255.0 * orig_data, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    )
    xt = torch.clamp(255.0 * xt, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    dist.barrier()
    samples_tensor = torch.as_tensor(
        samples, dtype=torch.uint8, device=device
    ).contiguous()
    sample_gather_list = [
        torch.empty_like(samples_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(sample_gather_list, samples_tensor)

    # get xt
    xt_tensor = torch.as_tensor(xt, dtype=torch.uint8, device=device).contiguous()
    xt_gather_list = [torch.empty_like(xt_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(xt_gather_list, xt_tensor)

    # get original data
    orig_data_tensor = torch.as_tensor(
        orig_data, dtype=torch.uint8, device=device
    ).contiguous()
    orig_data_gather_list = [
        torch.empty_like(orig_data_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(orig_data_gather_list, orig_data_tensor)

    labels_tensor = torch.as_tensor(
        class_labels, dtype=torch.int64, device=device
    ).contiguous()
    label_gather_list = [
        torch.empty_like(labels_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(label_gather_list, labels_tensor)

    if rank == 0:
        gathered = torch.cat(sample_gather_list, dim=0)[: cfg.num_samples]
        gathered_xt = torch.cat(xt_gather_list, dim=0)[: cfg.num_samples]
        gathered_orig = torch.cat(orig_data_gather_list, dim=0)[: cfg.num_samples]

        posterior_samples_path = f"{sample_folder_dir}/posterior_samples.npy"
        np.save(posterior_samples_path, gathered.to("cpu", dtype=torch.uint8).numpy())

        original_data_path = f"{sample_folder_dir}/original_data.npy"
        np.save(original_data_path, gathered_orig.to("cpu", dtype=torch.uint8).numpy())

        xt_data_path = f"{sample_folder_dir}/xt_data.npy"
        np.save(xt_data_path, gathered_xt.to("cpu", dtype=torch.uint8).numpy())

        print(f"OUTPUT_PATH:{posterior_samples_path}")

        gathered_labels = torch.cat(label_gather_list, dim=0)[: cfg.num_samples]
        labels_npy_path = f"{sample_folder_dir}/labels.npy"
        np.save(
            labels_npy_path, gathered_labels.to("cpu", dtype=torch.int64).numpy()
        )


if __name__ == "__main__":
    main()
