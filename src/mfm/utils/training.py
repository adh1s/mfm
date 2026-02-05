import copy
import gc
import math
import os
import re

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import tqdm
from diffusers import AutoencoderKL
from lightning.pytorch.callbacks import Callback, WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

import wandb
from mfm.SI.samplers import kernel_sampler_fn
from mfm.utils.ema import EMAWeightAveraging
from mfm.utils.evaluation import (plot_posterior_samples,
                                      posterior_sampling_fn)


def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, model, weighting_model, loss_fn, SI):
        super().__init__()
        self.model = model
        self._teacher_container = []
        
        if cfg.loss.distill_fm:
            teacher_model = copy.deepcopy(self.model)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            self._teacher_container.append(teacher_model)
        
        if cfg.loss.data_fm:
            self.weighting_model = weighting_model
        else:
            self.weighting_model = None
        
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.SI = SI

        self._vae_container = []

        print("Initializing VAE for raw ImageNet training...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        self._vae_container.append(vae)
        self.register_buffer('latents_scale', torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1))
        self.register_buffer('latents_bias', torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1))

    @property
    def vae(self):
        return self._vae_container[0] if self._vae_container else None
    
    @property
    def teacher_model(self):
        return self._teacher_container[0] if self._teacher_container else None

    def forward(self, x):
        return self.model(x)

    def setup(self, stage: str):
        self.vae.to(self.device)

        if self.teacher_model:
            self.teacher_model.to(self.device)
        if self.weighting_model:
            self.weighting_model.to(self.device)

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
            with torch.amp.autocast('cuda', enabled=False):
                x_vae = x.to(dtype=self.vae.dtype)
                latents = self.vae.encode(x_vae).latent_dist.sample()
                latents = (latents - self.latents_bias) * self.latents_scale
                x = latents.to(dtype=x.dtype) # Cast back to original dtype (likely bf16)

        losses, aux_losses = self.loss_fn(self.model, self.weighting_model, x, labels, step, 
                                          teacher_model=self.teacher_model)
        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        for name, loss in aux_losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        total_loss = 0
        
        for name, loss in losses.items():
            total_loss += loss

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        return total_loss

    def on_before_optimizer_step(self, optimizer,):
        if self.global_step % 10 == 0:
            total_norm = torch.norm(torch.stack([
                p.grad.detach().norm(2)
                for p in self.parameters() if p.grad is not None
            ]))
            self.log("grad_l2_norm", total_norm, on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """Run validation with EMA parameters"""
        x, labels = batch
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                x_vae = x.to(dtype=self.vae.dtype)
                latents = self.vae.encode(x_vae).latent_dist.sample()
                latents = (latents - self.latents_bias) * self.latents_scale
                x = latents.to(dtype=x.dtype)

        ema_val_loss, ema_val_aux_losses = self.loss_fn(self.model, self.weighting_model, x, labels, self.global_step, teacher_model=self.teacher_model)
        for name, loss in ema_val_loss.items():
            self.log(f"val_ema/{name}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for name, loss in ema_val_aux_losses.items():
            self.log(f"val_ema/{name}", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        ema_total_loss = sum(ema_val_loss.values())

        self.log("val_ema/total_loss", ema_total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return ema_total_loss
            
    def configure_optimizers(self):
        params = [p for p in self.parameters()]
        if self.cfg.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(params, lr=self.cfg.lr.val, weight_decay=self.cfg.get("weight_decay", 0.0))
        else:
            optimizer = torch.optim.Adam(params, lr=self.cfg.lr.val, weight_decay=self.cfg.get("weight_decay", 0.0))

        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, 
            factor=1.0, 
            total_iters=float('inf'))

        if self.cfg.lr.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.cfg.lr.warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.cfg.lr.warmup_steps]
            )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

class SamplingCallback(Callback):
    """Callback for generating and saving samples during training"""
    def __init__(self, cfg, test_data, test_labels, inverse_scaler, SI):
        super().__init__()
        self.cfg = cfg
        self.test_data = test_data
        self.test_labels = test_labels 
        self.inverse_scaler = inverse_scaler
        self.SI = SI
        # Use model input shape (latent shape) for sampling
        self.image_shape = (
            self.cfg.model.in_channels, 
            self.cfg.model.input_size, 
            self.cfg.model.input_size
        )
        self._last_sampled_step = -1 
        
    def setup(self, trainer, pl_module, stage=None):
        self.world_size = trainer.world_size
        self.rank = trainer.global_rank
        device = pl_module.device

        # Make unconditional samples and n_conditioning_samples divisible by world size
        self.cfg.sampling.n_unconditional_samples = (
            self.cfg.sampling.n_unconditional_samples // self.world_size
        ) * self.world_size
        self.cfg.sampling.n_conditioning_samples = (
            self.cfg.sampling.n_conditioning_samples // self.world_size
        ) * self.world_size

        g = torch.Generator(device='cpu')
        g.manual_seed(self.cfg.seed + 12345) # Fixed seed

        self.shared_unconditional_labels = torch.randint(
            low=0,
            high=self.cfg.model.label_dim,
            size=(self.cfg.sampling.n_unconditional_samples,),
            generator=g,
            device='cpu',
        ).to(device)

        self.shared_unconditional_noise = self._create_shared_tensor(
            (self.cfg.sampling.n_unconditional_samples, *self.image_shape),
            g
        ).to(device)
        self.shared_noise = self._create_shared_tensor(
            (self.cfg.sampling.n_conditioning_samples, *self.image_shape),
            g
        ).to(device)
        self.shared_posterior_noise = self._create_shared_tensor(
            (self.cfg.sampling.n_conditioning_samples * self.cfg.sampling.n_samples_per_image, *self.image_shape),
            g
        ).to(device)
    def _create_shared_tensor(self, shape, generator):
        shared_tensor = torch.randn(
            shape,
            generator=generator,
            device=torch.device('cpu'),
        )
        return shared_tensor

    def _decode(self, pl_module, samples, vae_batch_size):
        # samples are scaled latents
        latents = samples
        
        # Handle 5D input [B, N, C, H, W]
        is_5d = latents.ndim == 5
        if is_5d:
            B, N, C, H, W = latents.shape
            latents = latents.view(B * N, C, H, W)

        latents = latents / pl_module.latents_scale + pl_module.latents_bias
        images = []
        
        for start in range(0, latents.shape[0], vae_batch_size):
            end = min(start + vae_batch_size, latents.shape[0])
            latents_batch = latents[start:end]
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=False):
                    latents_batch = latents_batch.to(dtype=pl_module.vae.dtype)
                    images_batch = pl_module.vae.decode(latents_batch).sample
                    images_batch = images_batch.to(dtype=latents.dtype) # Cast back
            images.append(images_batch)
        
        images = torch.cat(images, dim=0) 
        
        if is_5d:
            images = images.view(B, N, *images.shape[1:])

        images = self.inverse_scaler(images) # [0, 1]

        return images.clamp(0.0, 1.0)
    

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step) % self.cfg.sampling.every_n_steps != 0:
            return
        if (trainer.global_step == self._last_sampled_step):
            return 
    
        self._last_sampled_step = trainer.global_step
        was_training = pl_module.training
        pl_module.eval()
        self._run_distributed_sampling(pl_module, trainer) 
        if was_training:
            pl_module.train()
    
    def _gather(self, tensor, indices, trainer):
        """
        All-gather `tensor` across ranks and return a single concatenated tensor
        on every rank. Assumes the leading dim is batch.
        """
        if trainer.world_size == 1:
            return tensor

        gathered = trainer.strategy.all_gather(tensor)
        gathered = gathered.reshape(-1, *tensor.shape[1:])
        
        # Reorder to original order
        gathered_indices = trainer.strategy.all_gather(indices) 
        gathered_indices = gathered_indices.reshape(-1)
        order = torch.argsort(gathered_indices)
        gathered_tensor = gathered[order]

        return gathered_tensor
    
    def _run_distributed_sampling(self, pl_module, trainer):
        print("Running distributed sampling...")
        device = pl_module.device
        all_indices = torch.arange(self.cfg.sampling.n_unconditional_samples, device=device)

        # 1) unconditional sampling w/ CFG
        for n_steps in tqdm.tqdm(self.cfg.sampling.n_kernel_steps, desc="Unconditional sampling with CFG at different n_steps"):
            for cfg_scale in self.cfg.sampling.kernel_cfg_scales:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    device_indices = all_indices[self.rank::self.world_size]
                    unconditional_noise_device = self.shared_unconditional_noise[device_indices]
                    unconditional_labels_device = self.shared_unconditional_labels[device_indices]

                    # different seed per device (but fixed across runs)
                    generator = torch.Generator(device=device).manual_seed(self.cfg.seed + self.rank)
            
                    unconditional_samples_kernel = kernel_sampler_fn(
                            pl_module.model,
                            shape=self.image_shape,
                            shape_decoded=self.image_shape, 
                            SI=self.SI,
                            n_samples=unconditional_noise_device.shape[0],
                            n_batch_size=self.cfg.sampling.max_batch_size,
                            n_steps=n_steps,
                            inverse_scaler_fn=lambda x: x,
                            x0=unconditional_noise_device,
                            generator=generator,
                            class_labels=unconditional_labels_device,
                            cfg_scale=cfg_scale,
                        )
                
                unconditional_samples_kernel = self._decode(pl_module, unconditional_samples_kernel, vae_batch_size=self.cfg.sampling.vae_batch_size)
                
                unconditional_samples_kernel = self._gather(unconditional_samples_kernel, device_indices, trainer)
                if trainer.is_global_zero:
                    unconditional_samples_kernel = unconditional_samples_kernel.clamp(0.0, 1.0)
                    self._save_samples_unconditional(
                        pl_module, unconditional_samples_kernel, 
                        title=f"unconditional_samples_kernel_cfg_{cfg_scale}_{n_steps}")

        # get device dependent test-data/corruption-noise
        all_indices = torch.arange(self.cfg.sampling.n_conditioning_samples, device=device)
        device_indices = all_indices[self.rank::self.world_size]
        data_device = self.test_data[device_indices.cpu()].to(device)
        shared_noise_device = self.shared_noise[device_indices]
        
        # get init noise for posterior sampling
        all_indices = torch.arange(self.cfg.sampling.n_conditioning_samples * self.cfg.sampling.n_samples_per_image, 
                                    device=device)
        device_indices_init = all_indices[self.rank::self.world_size]
        shared_posterior_device = self.shared_posterior_noise[device_indices_init]
        labels_device = self.test_labels[device_indices.cpu()].to(device)

        # 2) Consistency sampling with model guidance
        for t_cond in tqdm.tqdm(self.cfg.sampling.consistency_sampler.t_conds, desc="Consistency sampling at different t_conds"):
            for cfg_scale in self.cfg.sampling.cfg_scales:
                for n_steps in self.cfg.sampling.consistency_sampler.steps_to_test:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        cons_x_t, cons_x_1 = self._sample_batch(
                            pl_module, t_cond, data_device, shared_noise_device, shared_posterior_device, 
                            labels=labels_device, cfg_scale=cfg_scale, posterior_sampler="consistency", n_steps=n_steps
                        )
                        cons_x_t = self._decode(pl_module, cons_x_t, vae_batch_size=self.cfg.sampling.vae_batch_size)
                        cons_x_1 = self._decode(pl_module, cons_x_1, vae_batch_size=self.cfg.sampling.vae_batch_size)

                    cons_x_t = self._gather(cons_x_t, device_indices, trainer)
                    cons_x_1 = self._gather(cons_x_1, device_indices, trainer)

                    if trainer.is_global_zero:
                        self._save_samples(pl_module, cons_x_t, cons_x_1, t_cond, f"consistency_cfg_{cfg_scale}", steps=n_steps)
        

    def _sample_batch(self, pl_module, t_cond, x1, noise_data, noise_start, labels=None, cfg_scale=None, x_cond_scale=None, **kwargs):
        x_t_list, x_1_list = [], []
        N_local = x1.shape[0]
        
        # effective batch size = bs * m \simeq max_batch_size
        bs = self.cfg.sampling.max_batch_size // self.cfg.sampling.n_samples_per_image 
        m = self.cfg.sampling.n_samples_per_image
        device = pl_module.device

        for start in tqdm.tqdm(range(0, N_local, bs), desc="Batches"):
            end = min(start + bs, N_local)
            cur_bs = end - start

            x1_batch = x1[start:end]
            noise_batch = noise_data[start:end]
    
            # Encode if VAE
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=False):
                    x1_batch = x1_batch.to(dtype=pl_module.vae.dtype)
                    latents = pl_module.vae.encode(x1_batch).latent_dist.sample()
                    latents = (latents - pl_module.latents_bias) * pl_module.latents_scale
                    x1_batch = latents.to(dtype=x1_batch.dtype) # Cast back

            # generate noisy input
            t_cond_batch = torch.full((cur_bs,), t_cond, device=device)
            alpha_t, beta_t = self.SI.get_coefficients(t_cond_batch) # Shape: [B,]
            alpha_t, beta_t = broadcast_to_shape(alpha_t, x1_batch.shape), broadcast_to_shape(beta_t, x1_batch.shape)
            xt_batch = alpha_t * noise_batch + beta_t * x1_batch # [cur_bs, C, H, W]
            
            # get starting point for sampler
            eps_start_batch = noise_start[start*m : end*m]  # [cur_bs*m, C, H, W]
            xt_batch = xt_batch.repeat_interleave(m, dim=0)  # [cur_bs*m, C, H, W]
            t_cond_batch = t_cond_batch.repeat_interleave(m, dim=0)  # [cur_bs*m,]
            
            labels_batch = labels[start:end]
            labels_batch = labels_batch.repeat_interleave(m, dim=0)  # [cur_bs*m,]
            cfg_batch = torch.full((cur_bs * m,), cfg_scale, device=device)

            model = pl_module.model if kwargs.get("v_type", None) != "glass_flows" else pl_module.teacher_model
            
            with torch.no_grad():
                xt, x1_out = posterior_sampling_fn(
                    model,
                    xt_batch,
                    t_cond_batch,
                    n_samples_per_image=m,
                    inverse_scaler=lambda x: x,
                    eps_start=eps_start_batch,
                    labels=labels_batch,
                    cfg_scales=cfg_batch,
                    x_cond_scales=None,
                    **kwargs
                )
            
            x_t_list.append(xt) # [cur_bs, m, C, H, W]
            x_1_list.append(x1_out) # [cur_bs, m, C, H, W]
        return torch.cat(x_t_list, dim=0), torch.cat(x_1_list, dim=0)
    
    def _save_samples(self, pl_module, x_t, x_0, t_cond, sample_type, steps=None):
        save_dir = os.path.join(self.cfg.work_dir, f"samples_{pl_module.global_step}_ema")
        os.makedirs(save_dir, exist_ok=True)

        # Generate file name
        steps_str = f"_steps_{steps}" if steps is not None else ""
        base_name = f"{sample_type}_samples_t_{t_cond}{steps_str}"
        
        # Save plot
        title = f"{sample_type.title()} Samples at t = {t_cond}"
        if steps is not None:
            title += f" with {steps} steps"

        save_dict = {
            'x_t': x_t.cpu(),
            'x_0_samples': x_0.cpu(),
            't_cond': t_cond,
        }
        if steps is not None:
            save_dict['n_steps'] = steps
                    
        fig = plot_posterior_samples(self.inverse_scaler(self.test_data[0:10].cpu().numpy()),
                                     x_t[0:10].cpu().numpy(),
                                     x_0[0:10, 0:10].cpu().numpy(),
                                     os.path.join(save_dir, f"{base_name}.png"),
                                     title)
        pl_module.logger.experiment.log({
            f"val/{base_name}_grid_ema": [wandb.Image(fig)],
            "global_step": pl_module.global_step
        })
        plt.close(fig)

    def _save_samples_unconditional(self, pl_module, samples, title):
        save_dir = os.path.join(self.cfg.work_dir, f"samples_{pl_module.global_step}_ema")
        os.makedirs(save_dir, exist_ok=True)

        base_name = f"unconditional_samples_{title}"
        
        N = samples.shape[0]
        nrow = math.ceil(math.sqrt(N))
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
        
        save_dict = {
            'samples': samples.cpu(),
            'sampler': title,
        }

        pl_module.logger.experiment.log({
            f"val/unconditional_grid_ema_{title}": [wandb.Image(grid)],
            "global_step": pl_module.global_step
        })

        torchvision.utils.save_image(
            grid,
            os.path.join(save_dir, f"{base_name}.png"),
        )
        