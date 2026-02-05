import json
import math
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision("high")
import torch.distributed as dist

from mfm.models.model_wrapper import SIModelWrapper
from mfm.utils.evaluation import set_seed
from mfm.utils.image_reward_utils import (get_image_reward_model,
                                              load_image_reward_fn)
from mfm.utils.steering import (euler_maruyama_sampler, euler_sampler,
                                    get_conditional_drift_fn,
                                    get_imagenet_vae_fn)


def save_images_individual(samples, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(samples.shape[0]):
        img = samples[idx].permute(1, 2, 0).cpu().numpy() * 255
        img_path = os.path.join(save_dir, f"sample_{idx:05d}.png")
        plt.imsave(img_path, img.astype(np.uint8))

datashapes = {
    "imagenet": (4, 32, 32),
}

@hydra.main(config_path="../conf/", config_name="config_sample_steering.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # set seed
    set_seed(cfg.seed)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = cfg.seed * dist.get_world_size() + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    model = instantiate(cfg.model)
    SI = instantiate(cfg.SI)
    model = SIModelWrapper(model, SI, cfg.use_parametrization)
    model.to(device)

    map_location = {"cuda:%d" % 0: "cuda:%d" % device}
    
    try:
        checkpoint = torch.load(cfg.checkpoint_path, map_location=map_location)
        checkpoint = {k[6:] if k.startswith("model.") else k: v for k, v in checkpoint.items()}
    except:
        checkpoint = torch.load(cfg.checkpoint_path, map_location=map_location)
        checkpoint = {"model."+k: v for k, v in checkpoint.items()}
    
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print("unexpected keys", unexpected_keys, "missing keys", missing_keys)
    
    # make output directory
    output_dir = Path(cfg.save_dir)
    # Generate timestamp on rank 0 and broadcast to all ranks
    if rank == 0:
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        time_tensor = torch.tensor([ord(c) for c in time], dtype=torch.long, device=device)
    else:
        time_tensor = torch.zeros(15, dtype=torch.long, device=device)  # timestamp is 15 chars
    
    dist.broadcast(time_tensor, src=0)
    time = ''.join([chr(int(c)) for c in time_tensor.cpu().tolist()])
    
    output_dir = output_dir / time
    print(f"Rank {rank} loaded checkpoint from {cfg.checkpoint_path}, saving to {output_dir}")
    
    # inverse scaler    
    print(f"Using, {cfg.image_reward.model_name} image reward model.")
    reward_fn = load_image_reward_fn(cfg.image_reward, device, cfg.image_reward.model_name)        
    _, inverse_scaler = get_imagenet_vae_fn(device)

    conditional_drift_fn = get_conditional_drift_fn(model, reward_fn, inverse_scaler,
                                                    type=cfg.sampler,
                                                    drift_estimator=cfg.drift_estimator, # dps, iwae, sne
                                                    mc_samples=cfg.mc_samples,
                                                    guidance_scale=cfg.guidance_scale, # mc samples for iwae/sne
                                                    renorm_gradient=cfg.get("renorm_gradient", False), # use for MFM-G
                                                    renorm_scale=cfg.get("renorm_scale", 1.0),
                                                    renorm_type=cfg.get("renorm_type", "rescale"))

    # Batching 
    if cfg.drift_estimator in ["iwae", "sne", "base"]:
        n = cfg.per_proc_batch_size // cfg.mc_samples
    else:
        n = cfg.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(cfg.num_samples / global_batch_size) * global_batch_size)
    
    # generator seed
    generator = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.randn(total_samples, *datashapes[cfg.dataset.name], device=device, generator=generator)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    
    if rank == 0:
        print("local batch size per GPU:", n, 
              "| global batch size:", global_batch_size,
              "| total samples to generate:", total_samples)
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    all_samples = []

    for batch_start in tqdm.tqdm(range(0, samples_needed_this_gpu, n), disable=(rank != 0)):
        current_batch_size = min(n, samples_needed_this_gpu - batch_start)
        model_kwargs = {"cfg_scale": torch.full((current_batch_size,), cfg.cfg_scale, device=device),
                        "class_labels": torch.full((current_batch_size,), cfg.class_label, device=device)}

        x0_batch = x0[rank * samples_needed_this_gpu + batch_start: rank * samples_needed_this_gpu + batch_start + current_batch_size]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if cfg.sampler == "ode":
                samples = euler_sampler(x0_batch, conditional_drift_fn,
                                        t_start=0.01,
                                        n_steps=cfg.n_steps,
                                        **model_kwargs)
            elif cfg.sampler == "sde":
                samples = euler_maruyama_sampler(x0_batch, conditional_drift_fn,
                                                t_start=0.01,
                                                n_steps=cfg.n_steps,
                                                **model_kwargs)

        torch.cuda.empty_cache()

        with torch.no_grad():
            samples = inverse_scaler(samples)
            samples = samples.clamp(0.0, 1.0)
        all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)[:samples_needed_this_gpu]
    
    torch.cuda.empty_cache()

    rewards = {'CLIP': [],
                'HPSv2': [],
                'PickScore': [],
                'ImageReward': []}
                
    BS = 64
    with torch.no_grad():
        for k in rewards.keys():
            rewards[k] = []
            reward_fn = get_image_reward_model(device, k)

            for i in range(0, all_samples.shape[0], BS):
                batch = all_samples[i:i+BS].to(device)

                score = reward_fn([cfg.image_reward.prompt]*batch.shape[0], batch)
                rewards[k].append(score)
            rewards[k] = torch.cat(rewards[k], dim=0)

    for k in rewards.keys():
        rewards[k] = rewards[k][:all_samples.shape[0]]
        print(f"Rank {rank}: Mean {k} score: {rewards[k].mean():.4f}")

    
    if cfg.get("shard", False): # save per rank
        rank_save_dir = output_dir / f"rank_{rank}"
        os.makedirs(rank_save_dir, exist_ok=True)
        save_path = os.path.join(rank_save_dir, "sampled_images.pt")
        torch.save(all_samples.cpu(), save_path)
        print(f"Rank {rank}: Saved {all_samples.shape[0]} samples to {save_path}")

        # Save rewards per rank too
        rewards_path = rank_save_dir / "sampled_image_rewards.pt"
        torch.save(rewards, rewards_path)
        print(f"Rank {rank}: Saved rewards to {rewards_path}")
        
        dist.barrier()
        dist.destroy_process_group()
        return
    else: # take all together > wait and gather
        dist.barrier()
        gather_list = [torch.zeros_like(all_samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_list, all_samples)
    
        # gather rewards
        gather_rewards = {}
        for key in rewards.keys():
            gather_rewards[key] = [torch.zeros_like(rewards[key]) for _ in range(dist.get_world_size())]
            dist.all_gather(gather_rewards[key], rewards[key])

    if rank == 0:
        all_samples = torch.cat(gather_list, dim=0)[:total_samples]
        save_images_individual(all_samples, output_dir / "individual_images")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "sampled_images.pt")
        torch.save(all_samples.cpu(), save_path)
        print(f"Saved sampled images to {save_path}")
    
        if not cfg.get("shard", False):
            fig, axes = plt.subplots(10, 10, figsize=(10, 10))
            for idx in range(100):
                i = idx // 10
                j = idx % 10
                if idx >= all_samples.shape[0]:
                    axes[i, j].axis('off')
                    continue
                img = all_samples[idx].permute(1, 2, 0).cpu().numpy() * 255
                axes[i, j].imshow(img.astype(np.uint8))
                axes[i, j].axis('off')

            plt.tight_layout()
            fig_path = output_dir / "sampled_grid.png"
            plt.savefig(fig_path)
            print(f"Saved sampled image grid to {fig_path}")

        all_rewards = {key: torch.cat(gather_rewards[key], dim=0)[:total_samples] for key in gather_rewards}
        rewards_path = output_dir / "sampled_image_rewards.pt"
        torch.save(all_rewards, rewards_path)           
        print(f"Saved sampled image rewards to {rewards_path}")
        
        json_path = output_dir / "metrics_and_prompts.json"
        with open(json_path, "w") as f:
            json.dump({
                "mean_rewards": {key: all_rewards[key].mean().item() for key in all_rewards},
                "prompt": cfg.image_reward.prompt,
                "negative_prompt": cfg.image_reward.negative_prompt if "negative_prompt" in cfg.image_reward else "",
                "cfg": OmegaConf.to_container(cfg, resolve=True)
            }, f, indent=4)
        print(f"Saved metrics and prompts to {json_path}")


if __name__ == "__main__":
    main()