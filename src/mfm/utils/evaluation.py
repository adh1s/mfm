import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import write_png

from mfm.SI.samplers import consistency_sampler_fn, ode_sampler_fn


def plot_posterior_samples(
    x_0,
    x_t,
    x_0_samples,
    save_path,
    title,
):
    def _to_numpy_channels_last(array):
        """Convert tensor/array in CHW to numpy HWC for plotting."""
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        if array.ndim == 3 and (
            array.shape[0] <= 4 and array.shape[0] != array.shape[-1]
        ):
            array = np.transpose(array, (1, 2, 0))
        return array

    x_0 = x_0.detach().cpu() if isinstance(x_0, torch.Tensor) else x_0
    x_t = x_t.detach().cpu() if isinstance(x_t, torch.Tensor) else x_t
    x_0_samples = (
        x_0_samples.detach().cpu()
        if isinstance(x_0_samples, torch.Tensor)
        else x_0_samples
    )

    x_0 = np.asarray(x_0)
    x_t = np.asarray(x_t)
    x_0_samples = np.asarray(x_0_samples)

    N, M, C = x_0_samples.shape[:3]
    cmap = "gray" if C == 1 else None

    f, axs = plt.subplots(N, M + 2, figsize=((M + 2) * 3, N * 3))
    if N == 1:
        axs = axs.reshape(1, -1)

    for i in range(N):
        axs[i, 0].imshow(_to_numpy_channels_last(x_0[i]), cmap=cmap)
        axs[i, 0].set_title("x_0")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(_to_numpy_channels_last(x_t[i]), cmap=cmap)
        axs[i, 1].set_title("x_t")
        axs[i, 1].axis("off")

        for j in range(M):
            axs[i, j + 2].imshow(_to_numpy_channels_last(x_0_samples[i, j]), cmap=cmap)
            axs[i, j + 2].set_title(f"x_0 sample {j + 1}")
            axs[i, j + 2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    return f



def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))


def posterior_sampling_fn(
    model,
    xt_cond,  # [B*N, C, H, W],
    t_cond,  # [B*N, 1]
    n_samples_per_image=4,
    inverse_scaler=lambda x: (x + 1) / 2,
    eps_start=None,
    labels=None,
    **kwargs,
):

    if "n_step" in kwargs:
        raise ValueError("n_step is deprecated, use n_steps instead.")
    if "cfg_scale" in kwargs:
        raise ValueError("cfg_scale is deprecated, use cfg_scales instead.")
    """Sample from the posterior distribution using the model."""
    posterior_sampler = kwargs.get("posterior_sampler", "consistency")
    if posterior_sampler == "consistency":
        x_sample = consistency_sampler_fn(
            model,
            xt_cond,
            t_cond=t_cond,
            n_steps=kwargs.get("n_steps", 1),
            eps_start=eps_start,
            class_labels=labels,
            cfg_scale=kwargs.get("cfg_scales", None),
        )
    elif posterior_sampler == "ode":
        x_sample = ode_sampler_fn(
            model,
            xt_cond,
            t_cond=t_cond,
            n_steps=kwargs.get("n_steps", 100),
            eps_start=eps_start,
            labels=labels,
            cfg_scales=kwargs.get("cfg_scales", None),
            x_cond_scales=kwargs.get("x_cond_scales", None),
            v_type=kwargs.get("v_type", "standard"),
            checkpoint_type=kwargs.get("checkpoint_type", "dmf"),
        )

    elif posterior_sampler == "distributional_diffusion":
        noise_population = torch.randn_like(xt_cond, device=xt_cond.device)
        x_sample = model(xt_cond, t_cond, noise_population)
    else:
        raise ValueError(f"Unknown posterior sampler: {posterior_sampler}")
    x_sample = x_sample.view(
        -1, n_samples_per_image, *x_sample.shape[1:]
    )  # [B, N, C, H, W]
    # [B, C, H, W], # [B, N, C, H, W]
    return inverse_scaler(xt_cond), inverse_scaler(x_sample)


def set_seed(seed: int):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
