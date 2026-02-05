import math
import os

import lightning as pl
import torch
import torch.distributed as dist
import torchvision
import tqdm
from diffusers import AutoencoderKL
from lightning.pytorch.callbacks import Callback
from timm.data import ImageNetInfo

import wandb


class FinetuningModule(pl.LightningModule):
    def __init__(
        self, cfg, model, base_model, mfm, reward_fn, loss_fn, SI, eval_reward_models={}
    ):
        super().__init__()
        self.model = model

        self.loss_fn = loss_fn
        self.cfg = cfg
        self.SI = SI

        self.teacher_model = base_model
        self.teacher_model.eval()
        self.mfm = mfm
        self.mfm.eval()
        self.reward_fn = reward_fn
        self.vae = None

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        self.vae = vae
        self.register_buffer(
            "latents_scale",
            torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1),
        )
        self.register_buffer(
            "latents_bias", torch.tensor([0.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1)
        )

        for p in self.teacher_model.parameters():
            p.requires_grad = False
        for p in self.mfm.parameters():
            p.requires_grad = False
        for p in self.reward_fn.parameters():
            p.requires_grad = False

        # additional models for evaluation
        self.eval_reward_models = eval_reward_models

        if self.eval_reward_models:
            for model in self.eval_reward_models.values():
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

        self.last_eval_step = -1

    def forward(self, x):
        return self.model(x)

    def setup(self, stage: str):
        if self.eval_reward_models:
            for model in self.eval_reward_models.values():
                model.to(self.device)

    def on_load_checkpoint(self, checkpoint):
        """
        Handle checkpoint loading for backward compatibility.
        """
        state_dict = checkpoint.get("state_dict", {})
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing_keys = model_keys - ckpt_keys

    def training_step(self, batch, batch_idx):
        x, labels = batch
        step = self.global_step

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                x_vae = x.to(dtype=self.vae.dtype)
                latents = self.vae.encode(x_vae).latent_dist.sample()
                latents = (latents - self.latents_bias) * self.latents_scale
                x = latents.to(
                    dtype=x.dtype
                )  # Cast back to original dtype (likely bf16)

        losses, aux_losses = self.loss_fn(
            self.model,
            self.teacher_model,
            self.mfm,
            self.reward_fn,
            x,
            labels,
            step,
            inverse_scaler_fn=self.inverse_scaler_fn,
        )
        for name, loss in losses.items():
            self.log(
                f"train/{name}",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
        for name, loss in aux_losses.items():
            self.log(
                f"train/{name}",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        total_loss = 0
        for name, loss in losses.items():
            total_loss += loss
        self.log(
            "train/total_loss",
            total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "train/lr",
            current_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        return total_loss

    def on_before_optimizer_step(
        self,
        optimizer,
    ):
        if self.global_step % 10 == 0:
            total_norm = torch.norm(
                torch.stack(
                    [
                        p.grad.detach().norm(2)
                        for p in self.parameters()
                        if p.grad is not None
                    ]
                )
            )
            self.log("grad_l2_norm", total_norm, on_step=True, prog_bar=True)

    def on_save_checkpoint(self, checkpoint):
        keys = list(checkpoint["state_dict"].keys())
        frozen_prefixes = ["teacher_model", "mfm", "reward_fn", "vae"]

        for key in keys:
            if any(key.startswith(f"{prefix}.") for prefix in frozen_prefixes):
                del checkpoint["state_dict"][key]

    def inverse_scaler_fn(self, x):
        decoded = self.vae.decode((x - self.latents_bias) / self.latents_scale).sample
        return (decoded + 1) / 2

    def validation_step(self, batch, batch_idx):
        """Run validation with EMA parameters"""
        return 0

    def configure_optimizers(self):
        params = [p for p in self.model.parameters()]
        if self.cfg.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(
                params,
                lr=self.cfg.lr.val,
                weight_decay=self.cfg.get("weight_decay", 0.0),
            )
        else:
            optimizer = torch.optim.Adam(
                params,
                lr=self.cfg.lr.val,
                weight_decay=self.cfg.get("weight_decay", 0.0),
            )

        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=float("inf")
        )

        if self.cfg.lr.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.cfg.lr.warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.cfg.lr.warmup_steps],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class RewardCallback(Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, trainer, pl_module, stage=None):
        self.world_size = trainer.world_size
        self.rank = trainer.global_rank
        # Use CPU generator to ensure consistent samples across ranks
        g = torch.Generator(device="cpu")
        g.manual_seed(self.cfg.seed + 12345)  # Fixed seed

        # Total samples global
        total_samples = self.cfg.sampling.num_samples
        self.noise = torch.randn(total_samples, 4, 32, 32, generator=g)

        if self.cfg.filter_for_classes:
            classes = self.cfg.filter_class_indices
            classes = torch.tensor(classes, device="cpu")
            random_indices = torch.randint(
                0, len(classes), (total_samples,), generator=g
            )
            self.labels = classes[random_indices]
        else:
            self.labels = torch.randint(0, 1000, (total_samples,), generator=g)

        samps_per_rank = math.ceil(total_samples / self.world_size)
        # Partition for current rank using slicing
        start_idx = self.rank * samps_per_rank
        end_idx = start_idx + samps_per_rank

        self.local_noise = self.noise[start_idx:end_idx]
        self.local_labels = self.labels[start_idx:end_idx]

    def inverse_scaler_fn(self, pl_module, x):
        decoded = pl_module.vae.decode(
            (x - pl_module.latents_bias) / pl_module.latents_scale
        ).sample
        return (decoded + 1) / 2

    def on_train_start(self, trainer, pl_module):
        self.evaluate_reward(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.evaluate_reward(pl_module)

    @torch.no_grad()
    def evaluate_reward(self, pl_module):
        STEPS = 250
        BS = 64

        # Keep data on CPU, move to device in batches
        x_gen_cpu = self.local_noise
        labels_gen_cpu = self.local_labels
        NUM_SAMPLES = x_gen_cpu.shape[0]
        dt = 1.0 / STEPS

        class_description = ImageNetInfo()

        # Store samples on CPU to save VRAM
        samples_list, rewards_list = [], []
        additional_metrics = {}

        for batch_start in range(0, NUM_SAMPLES, BS):
            # Move only the current batch to GPU
            x_batch = x_gen_cpu[batch_start : batch_start + BS].to(pl_module.device)
            labels_batch = labels_gen_cpu[batch_start : batch_start + BS].to(
                pl_module.device
            )

            x_temp = x_batch
            for i in range(STEPS):
                t = i * dt
                t_batch = torch.full((x_temp.shape[0],), t, device=x_temp.device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    drift = pl_module.model(
                        x_temp,
                        t_batch,
                        labels_batch,
                        cfg_scale=torch.full_like(
                            t_batch, self.cfg.get("cfg_scale", 2.5)
                        ),
                    )
                x_temp = x_temp + drift * dt

            # Use pl_module when calling inverse_scaler_fn
            batch_samples = self.inverse_scaler_fn(pl_module, x_temp)
            prompts = []

            if self.cfg.loss.get("prompt_type", "auto") == "auto":
                for label in labels_batch:
                    label = label.int()
                    class_desc = class_description.index_to_description(label)
                    prompts.append(f"{self.cfg.loss.prompt_prefix}{class_desc}")
            else:
                prompts.append(self.cfg.loss.prompt)

            batch_rewards = pl_module.reward_fn.score(batch_samples, prompts)
            if pl_module.eval_reward_models:
                for name, eval_model in pl_module.eval_reward_models.items():
                    eval_rewards = eval_model.score(batch_samples, prompts)
                    if name not in additional_metrics:
                        additional_metrics[name] = []
                    additional_metrics[name].append(eval_rewards)

            rewards_list.append(batch_rewards)
            # Move samples to CPU immediately
            samples_list.append(batch_samples.cpu())

        samples = torch.cat(samples_list, dim=0)
        rewards = torch.cat(rewards_list, dim=0)

        all_rewards = pl_module.all_gather(rewards)

        if all_rewards.ndim > 1:
            all_rewards = all_rewards.view(-1)

        n_vis_per_rank = min(64, samples.shape[0])  # e.g. 8 * 8 GPUs = 64 images
        vis_samples = samples[:n_vis_per_rank].to(
            pl_module.device
        )  # Move back to GPU for all_gather

        all_vis_samples = pl_module.all_gather(vis_samples)

        if all_vis_samples.ndim > 4:
            all_vis_samples = all_vis_samples.view(-1, *all_vis_samples.shape[2:])

        # Handle additional metrics
        for name, metrics_list in additional_metrics.items():
            metrics = torch.cat(metrics_list, dim=0)
            all_metrics = pl_module.all_gather(metrics)
            if all_metrics.ndim > 1:
                all_metrics = all_metrics.view(-1)

            if self.rank == 0:
                mean_metric = all_metrics.mean().item()
                print(f"Mean {name} of generated samples: {mean_metric}")
                # pl_module.log(f"gen/mean_{name}", mean_metric, prog_bar=True)
                if pl_module.logger:
                    pl_module.logger.experiment.log(
                        {
                            f"gen/mean_{name}": mean_metric,
                            "global_step": pl_module.global_step,
                        }
                    )

        if self.rank == 0:
            print("Mean reward of generated samples:", all_rewards.mean().item())
            print(
                "Min reward of generated samples:",
                all_rewards.min().item(),
                "Max reward of generated samples:",
                all_rewards.max().item(),
            )

            # pl_module.log("gen/mean_reward", all_rewards.mean().item(), prog_bar=True)
            os.makedirs(f"{self.cfg.work_dir}/samples", exist_ok=True)
            save_name = f"{self.cfg.work_dir}/samples/val_{pl_module.global_step}.png"

            # Use the gathered visualization samples
            all_vis_samples = all_vis_samples.clamp(0, 1)
            all_vis_samples = all_vis_samples.to(torch.float32)

            grid_nrow = int(math.sqrt(all_vis_samples.shape[0]))
            torchvision.utils.save_image(all_vis_samples, save_name, nrow=grid_nrow)

            # log to wandb
            grid = torchvision.utils.make_grid(all_vis_samples, nrow=grid_nrow)
            if pl_module.logger:
                pl_module.logger.experiment.log(
                    {
                        "val/generations": [wandb.Image(grid)],
                        "gen/mean_reward": all_rewards.mean().item(),
                        "global_step": pl_module.global_step,
                    }
                )

        return
