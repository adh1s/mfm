import math
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
import torch.distributed as dist
import torch.nn as nn
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader, distributed

from mfm.data import get_data_module
from mfm.models.model_wrapper import SIModelWrapper
from mfm.utils.evaluation import posterior_sampling_fn
from mfm.utils.image_reward_utils import rm_load
from mfm.utils.steering import get_imagenet_vae_fn


def sigma_t_sq(t):
    return 2 * torch.clamp((1 / (t + 1e-8) - 1), min=0, max=25)

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

@torch.no_grad()
def euler_ddpm_sampler(x_t, t, model, n_steps=100, **model_kwargs):
    N = x_t.shape[0]
    x = x_t
    t_start = t[0]
    t_steps = torch.linspace(t_start, 1.0, n_steps + 1, device=x.device)

    for i in tqdm(range(n_steps), desc="DDPM (ground-truth) sampling"):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        dt = t_next - t_cur
        t_batched = torch.full((N,), t_cur, device=x.device)
        v = model.v(
            t_batched,
            t_batched,
            x,
            torch.zeros_like(t_batched),
            torch.zeros_like(x),
            **model_kwargs,
        )
        drift = 2 * v - (1 / t_cur) * x
        diffusion = torch.sqrt(sigma_t_sq(t_cur))
        x = x + drift * dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)
    return x


@hydra.main(
    config_path="../conf/", config_name="config_sample_value.yaml", version_base="1.3"
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    ckpt_path = cfg.checkpoint_path

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = cfg.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    time_string = time.strftime("%Y%m%d-%H%M%S")

    if cfg.save_ground_truth:
        name = f"ddpm_ground_truth_{cfg.ddpm.n_steps}_{cfg.ddpm.n_samples}_t_{str(cfg.posterior_t).replace('.', '_')}_{time_string}"
        name += cfg.reward_prompt.replace(" ", "_").replace(",", "").replace(".", "")
        log_dir = Path(cfg.ground_truth_path) / f"{name}"
    else:
        name = f"value_function_estimation_k_{cfg.n_samples_mc}_n_{cfg.n_steps_mc}_t_{str(cfg.posterior_t).replace('.', '_')}_{cfg.n_method_mc}_{time_string}"
        name += cfg.reward_prompt.replace(" ", "_").replace(",", "").replace(".", "")
        log_dir = Path(ckpt_path).parent / f"{name}"

    cfg.work_dir = str(log_dir)
    print(f"Logging to {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    sample_folder_dir = log_dir / "samples"
    os.makedirs(sample_folder_dir, exist_ok=True)

    # instantiate model
    model = instantiate(cfg.model)
    SI = instantiate(cfg.SI)
    model = SIModelWrapper(model, SI, cfg.use_parametrization)
    model.to(device)

    # load checkpoint
    map_location = {"cuda:%d" % 0: "cuda:%d" % device}

    if cfg.init_from_dmf:
        print("Initializing from DMF checkpoint...")
        sit_state_dict = torch.load(cfg.dmf_checkpoint_path, map_location=map_location)
        sit_state_dict = {
            k.replace("module.", ""): v for k, v in sit_state_dict.items()
        }
        # Target model (DiT)
        target_model = model.model
        if hasattr(target_model, "dit"):
            target_model = target_model.dit

        # Load state dict
        print("Loading state dict into DiT...")
        missing, unexpected = target_model.load_state_dict(sit_state_dict, strict=False)
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

        # Initialize s_embedder from t_embedder (checkpoint)
        print("Initializing s_embedder from t_embedder...")
        target_model.s_embedder.load_state_dict(target_model.t_embedder.state_dict())
        target_model.t_embedder_second.load_state_dict(
            target_model.t_embedder.state_dict()
        )
        # zero t_embedder
        nn.init.constant_(target_model.t_embedder.mlp[2].weight, 0)
        nn.init.constant_(target_model.t_embedder.mlp[2].bias, 0)
    else:
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        checkpoint = {
            k[6:] if k.startswith("model.") else k: v for k, v in checkpoint.items()
        }
        missing, _ = model.load_state_dict(checkpoint, strict=False)
        print(f"Missing keys: {len(missing)}")

    model.eval()

    # LOAD IN VAE
    _, decode_fn = get_imagenet_vae_fn(device)

    # LOAD IN REWARD MODEL
    reward_model = rm_load("ImageReward-v1.0", device=device)

    def reward_function(images, prompt=cfg.reward_prompt):
        imgs = [img for img in images]
        prompts = [prompt] * len(images)
        rewards = reward_model.score_from_prompt_batched(prompts=prompts, images=imgs)
        rewards = torch.tensor(rewards, device=images.device)
        return rewards

    samples_needed_this_gpu = math.ceil(cfg.num_samples / dist.get_world_size())

    image_shape = (3, cfg.dataset.img_resolution, cfg.dataset.img_resolution)

    # get data module!
    datamodule = get_data_module(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")
    orig_loader = datamodule.test_dataloader()

    dataset = orig_loader.dataset
    sampler = distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=getattr(orig_loader, "shuffle", True),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_proc_batch_size,
        sampler=sampler,
        num_workers=getattr(orig_loader, "num_workers", 1),
        pin_memory=True,
        collate_fn=getattr(orig_loader, "collate_fn", None),
        drop_last=True,
    )

    # original and noised data
    orig_data = torch.zeros((samples_needed_this_gpu, *image_shape), device=device)
    noise_data = torch.zeros((samples_needed_this_gpu, *image_shape), device=device)

    if cfg.save_ground_truth:
        n_samples_per_image = cfg.ddpm.n_samples
    else:
        n_samples_per_image = cfg.n_samples_mc

    # store rewards + posterior samples
    samples = torch.zeros(
        (samples_needed_this_gpu, n_samples_per_image, *image_shape), device=device
    )
    rewards = torch.zeros((samples_needed_this_gpu,), device=device)
    rewards_all = torch.zeros(
        (samples_needed_this_gpu, n_samples_per_image), device=device
    )

    n_samples = 0

    for _, batch in tqdm(enumerate(dataloader), desc=f"Rank {rank} Sampling Batches"):
        if n_samples >= samples_needed_this_gpu:
            break

        x_full, y_full = batch
        x_full = x_full.to(device=device)
        y_full = y_full.to(device=device)

        # filter for class of interest
        if cfg.cls is not None:
            x_filter = x_full[y_full == cfg.cls]
            y_filter = y_full[y_full == cfg.cls]
            if x_filter.shape[0] == 0:
                continue
        else:
            x_filter = x_full
            y_filter = y_full

        for x, y in zip(x_filter, y_filter):
            if n_samples >= samples_needed_this_gpu:
                break

            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            orig_data[n_samples : n_samples + 1] = x

            # encode
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=False):
                    x_vae = x.to(dtype=vae.dtype)
                    latents = vae.encode(x_vae).latent_dist.sample()
                    latents = (latents - latents_bias) * latents_scale
                    x = latents.to(
                        dtype=x.dtype
                    )  # Cast back to original dtype (likely bf16)

            # sample noisy latent/store
            noise = torch.randn_like(x)
            xt = noise * (1 - cfg.posterior_t) + x * cfg.posterior_t
            t_cond = torch.full((x.shape[0],), cfg.posterior_t, device=device)
            noise_data[n_samples : n_samples + 1] = decode_fn(xt)

            # prepare labels for cfg
            labels = y.to(device=device)
            cfg_scale = torch.full((x.shape[0],), cfg.cfg_scale, device=device)

            x0_samples_batch, rewards_batch = [], []
            batches = math.ceil(n_samples_per_image / cfg.per_proc_batch_size)

            for i in range(batches):
                start_idx = i * cfg.per_proc_batch_size
                end_idx = min((i + 1) * cfg.per_proc_batch_size, n_samples_per_image)

                # number of posterior samples in this mini-batch
                curr_bs = end_idx - start_idx

                # repeat for batching
                xt_batch = xt.repeat_interleave(curr_bs, dim=0)
                t_cond_batch = t_cond.repeat_interleave(curr_bs, dim=0)
                labels_batch = labels.repeat_interleave(curr_bs, dim=0)
                cfg_scale_batch = cfg_scale.repeat_interleave(curr_bs, dim=0)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        # sde rollouts!
                        if cfg.save_ground_truth:
                            x0 = euler_ddpm_sampler(
                                xt_batch,
                                t_cond_batch,
                                model,
                                n_steps=cfg.ddpm.n_steps,
                                class_labels=labels_batch,
                                cfg_scale=cfg_scale_batch,
                            )

                            x0 = decode_fn(x0)
                        else:
                            x0 = posterior_sampling_fn(
                                model,
                                xt_batch,
                                t_cond_batch,
                                n_samples_per_image=curr_bs,
                                inverse_scaler=decode_fn,
                                labels=labels_batch,
                                cfg_scale=cfg_scale_batch,
                                posterior_sampler=cfg.n_method_mc,
                                n_steps=cfg.n_steps_mc,
                                v_type=cfg.get("posterior_v_type", "glass_flows"),
                            )[1]

                            x0 = x0.reshape(-1, *x0.shape[2:])

                        # save and compute rewards
                        x0_samples_batch.append(x0)
                        rewards_minibatch = reward_function(
                            x0, prompt=cfg.reward_prompt
                        )
                        rewards_batch.append(rewards_minibatch)

            x0_samples_batch = torch.cat(x0_samples_batch, dim=0)
            rewards_batch = torch.cat(rewards_batch, dim=0)

            x0_samples_batch = x0_samples_batch[:n_samples_per_image]
            rewards_batch = rewards_batch[:n_samples_per_image]
            value_estimate = torch.logsumexp(rewards_batch, dim=0) - math.log(
                n_samples_per_image
            )

            samples[n_samples : n_samples + 1, : cfg.ddpm.n_samples] = (
                x0_samples_batch.unsqueeze(0)
            )
            rewards_all[n_samples : n_samples + 1, : cfg.n_samples_mc] = (
                rewards_batch.unsqueeze(0)
            )
            rewards[n_samples : n_samples + 1] = value_estimate
            n_samples += 1

    # brodcast and gather all data
    orig_data = (
        torch.clamp(255.0 * orig_data, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    )
    noise_data = (
        torch.clamp(255.0 * noise_data, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    )
    samples = torch.clamp(255.0 * samples, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    dist.barrier()

    noise_data_tensor = torch.as_tensor(
        noise_data, dtype=torch.uint8, device=device
    ).contiguous()
    noise_data_gather_list = [
        torch.empty_like(noise_data_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(noise_data_gather_list, noise_data_tensor)

    orig_data_tensor = torch.as_tensor(
        orig_data, dtype=torch.uint8, device=device
    ).contiguous()
    orig_data_gather_list = [
        torch.empty_like(orig_data_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(orig_data_gather_list, orig_data_tensor)

    samples_tensor = torch.as_tensor(
        samples, dtype=torch.uint8, device=device
    ).contiguous()
    samples_gather_list = [
        torch.empty_like(samples_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(samples_gather_list, samples_tensor)

    rewards_tensor = torch.as_tensor(
        rewards, dtype=torch.float32, device=device
    ).contiguous()
    rewards_gather_list = [
        torch.empty_like(rewards_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(rewards_gather_list, rewards_tensor)

    rewards_all_tensor = torch.as_tensor(
        rewards_all, dtype=torch.float32, device=device
    ).contiguous()
    rewards_all_gather_list = [
        torch.empty_like(rewards_all_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(rewards_all_gather_list, rewards_all_tensor)

    if rank == 0:
        gathered_noise_data = torch.cat(noise_data_gather_list, dim=0)[
            : cfg.num_samples
        ]
        np.save(
            f"{sample_folder_dir}/noise_data.npy",
            gathered_noise_data.to("cpu", dtype=torch.uint8).numpy(),
        )
        gathered_orig = torch.cat(orig_data_gather_list, dim=0)[: cfg.num_samples]
        np.save(
            f"{sample_folder_dir}/original_data.npy",
            gathered_orig.to("cpu", dtype=torch.uint8).numpy(),
        )

        gathered_rewards = (
            torch.cat(rewards_gather_list, dim=0)[: cfg.num_samples].to("cpu").numpy()
        )
        np.save(f"{sample_folder_dir}/rewards.npy", gathered_rewards)
        gathered_samples = torch.cat(samples_gather_list, dim=0)[: cfg.num_samples]
        np.save(
            f"{sample_folder_dir}/posterior_samples.npy",
            gathered_samples.to("cpu", dtype=torch.uint8).numpy(),
        )

        # save all rewards
        gathered_rewards_all = (
            torch.cat(rewards_all_gather_list, dim=0)[: cfg.num_samples]
            .to("cpu")
            .numpy()
        )
        np.save(f"{sample_folder_dir}/all_rewards.npy", gathered_rewards_all)

        print(f"OUTPUT_PATH:{sample_folder_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
