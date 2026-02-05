import math

import torch
from diffusers import AutoencoderKL
from tqdm import tqdm

from mfm.utils.evaluation import posterior_sampling_fn


def get_imagenet_vae_fn(device):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    latents_scale = (
        torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
    )
    latents_bias = torch.tensor([0.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1).to(device)
    vae.eval()

    def decode_fn(x):
        n_dim = x.ndim
        if n_dim == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
        decoded = vae.decode((x - latents_bias) / latents_scale).sample
        decoded = (decoded + 1) / 2.0
        decoded = decoded.clamp(0.0, 1.0)
        
        if n_dim == 5:
            decoded = decoded.view(B, N, *decoded.shape[1:])

        return decoded

    def encode_fn(x):
        with torch.amp.autocast("cuda", enabled=False):
            x_vae = x.to(dtype=vae.dtype)
            latents = vae.encode(x_vae).latent_dist.sample()
            latents = (latents - latents_bias) * latents_scale
            x = latents.to(dtype=x.dtype)
        return x

    return encode_fn, decode_fn


def broadcast_to_shape(tensor, target):
    """Reshape tensor to broadcast with target by adding trailing singleton dims."""
    target_shape = target.shape if isinstance(target, torch.Tensor) else target
    extra_dims = len(target_shape) - tensor.dim()
    if extra_dims < 0:
        raise ValueError(
            f"Cannot broadcast: tensor has {tensor.dim()} dims, target has {len(target_shape)}"
        )
    view_shape = list(tensor.shape) + [1] * extra_dims
    return tensor.view(*view_shape)


def batched_reward_estimation(
    model, x, t, reward_fn, inverse_scaler, mc_samples=4, **model_kwargs
):
    B = x.shape[0]
    x_batched = x.repeat_interleave(mc_samples, dim=0)  # [N*B, C, H, W]
    t_batched = t.repeat_interleave(mc_samples, dim=0)  # [N*B]
    kw_rep = {
        k: v.repeat_interleave(mc_samples, dim=0) if isinstance(v, torch.Tensor) else v
        for k, v in model_kwargs.items()
    }

    x0_flow = torch.randn_like(x_batched)  # [N*B, C, H, W]
    t0_flow = torch.zeros_like(t_batched)  # [N*B]
    t1_flow = torch.ones_like(t_batched)  # [N*B]
    vst = model.v(t0_flow, t1_flow, x0_flow, t_batched, x_batched, **kw_rep)
    x1_samples = model.X(t0_flow, t1_flow, x0_flow, vst)  # [N*B, C, H, W]
    x1_data = inverse_scaler(x1_samples)  # [N*B, C, H, W]
    reward = reward_fn(x1_data)
    reward = reward.view(B, mc_samples)  # [B, N]
    x1_samples = x1_samples.view(
        B, mc_samples, *x1_samples.shape[1:]
    )  # [B, N, C, H, W]
    x1_data = x1_data.view(B, mc_samples, *x1_data.shape[1:])  # [B, N, C, H, W]
    x_batched = x_batched.view(B, mc_samples, *x_batched.shape[1:])  # [B, N, C, H, W]
    return {
        "x1_samples": x1_samples,
        "reward": reward,
        "x1_data": x1_data,
        "xt": x_batched,
    }


def get_dps_score_fn(model, reward_fn, inverse_scaler):
    def score_fn(x, t, v_pred, **model_kwargs):
        B = x.shape[0]  # x1
        x1_pred = get_tweedie_estimate(x, t, v_pred)  # [B, C, H, W]
        x1_data = inverse_scaler(x1_pred)  # [B, C, H, W]
        reward = reward_fn(x1_data)
        grad = torch.autograd.grad(reward.sum(), x)[0]
        return grad

    return score_fn


def get_iwae_score_fn(model, reward_fn, inverse_scaler, mc_samples=4):
    def score_fn(x, t, **model_kwargs):
        x = x.detach().requires_grad_(True)  # Enable gradient track
        ret = batched_reward_estimation(
            model, x, t, reward_fn, inverse_scaler, mc_samples, **model_kwargs
        )  # [B, N]
        reward = ret["reward"]  # [B, N]
        iwae = torch.logsumexp(reward, dim=1) - torch.log(
            torch.tensor(mc_samples, dtype=reward.dtype, device=reward.device)
        )
        iwae = iwae.sum()  # [B,]
        grad = torch.autograd.grad(iwae, x)[0]
        return grad

    return score_fn


def get_sne_score_fn(model, reward_fn, inverse_scaler, mc_samples=4):
    def score_fn(x, t, base_drift, **model_kwargs):
        with torch.no_grad():
            B = x.shape[0]
            with torch.no_grad():
                ret = batched_reward_estimation(
                    model, x, t, reward_fn, inverse_scaler, mc_samples, **model_kwargs
                )  # [B, N]
            reward, x1_samples = ret["reward"], ret["x1_samples"]
            weights = torch.softmax(reward, dim=1)  # [B, N]
            weights_view = broadcast_to_shape(weights, x1_samples.shape)
            x1_twisted = (weights_view * x1_samples).sum(dim=1)

            x1_uniform = x1_samples.mean(dim=1)
            t_denom = broadcast_to_shape(1.0 - t, x1_twisted.shape).clamp(min=1e-3)
            correction = (x1_twisted - x1_uniform) / t_denom
        return correction

    return score_fn


def sigma_t_sq(t):
    return 2 * torch.clamp((1 / (t + 1e-8) - 1), min=0, max=25)


def get_unconditional_drift_sde(model, grad=False):
    context = torch.no_grad() if not grad else torch.enable_grad()

    def drift_fn(x, t, **model_kwargs):
        with context:
            v = model.v(
                t, t, x, torch.zeros_like(t), torch.zeros_like(x), **model_kwargs
            )
        t = broadcast_to_shape(t, x.shape)
        drift = 2 * v - (1 / t) * x
        return {"drift": drift, "v": v}

    return drift_fn


def get_unconditional_drift_ode(model, grad=False):
    context = torch.no_grad() if not grad else torch.enable_grad()

    def drift_fn(x, t, **model_kwargs):
        with context:
            v = model.v(
                t, t, x, torch.zeros_like(t), torch.zeros_like(x), **model_kwargs
            )
        return {"drift": v, "v": v}

    return drift_fn


def get_tweedie_estimate(x, t, v_pred):
    broadcast_time = broadcast_to_shape(1 - t, x.shape)
    x1_pred = x + broadcast_time * v_pred
    return x1_pred


def get_conditional_drift_fn(
    model,
    reward_fn,
    inverse_scaler,
    type="ode",
    drift_estimator="dps",
    mc_samples=4,
    guidance_scale=1.0,
    renorm_gradient=False,
    renorm_scale=1.0,
    renorm_type="rescale",
):
    grad = True if drift_estimator == "dps" else False
    if type == "sde":
        unconditional_drift_fn = get_unconditional_drift_sde(model, grad=grad)
    elif type == "ode":
        unconditional_drift_fn = get_unconditional_drift_ode(model, grad=grad)
    else:
        raise ValueError(f"Unknown drift type: {type}")

    if drift_estimator == "dps":
        steering_drift_fn = get_dps_score_fn(model, reward_fn, inverse_scaler)
    elif drift_estimator == "iwae":
        steering_drift_fn = get_iwae_score_fn(
            model, reward_fn, inverse_scaler, mc_samples=mc_samples
        )
    elif drift_estimator == "sne":
        steering_drift_fn = get_sne_score_fn(
            model, reward_fn, inverse_scaler, mc_samples=mc_samples
        )
    elif drift_estimator == "base":
        steering_drift_fn = lambda x, *args, **kwargs: torch.zeros_like(x)
    else:
        raise ValueError(f"Unknown drift estimator: {drift_estimator}")

    def drift_fn(x, t, **model_kwargs):
        if drift_estimator == "dps":  # track gradients through
            x = x.detach().requires_grad_(True)
        drift_ret = unconditional_drift_fn(x, t, **model_kwargs)

        # get some quantities
        base_drift = drift_ret["drift"]
        v_pred = drift_ret["v"]
        tweedie_estimate = get_tweedie_estimate(x, t, v_pred)

        with torch.no_grad():
            tweedie_estimate = inverse_scaler(tweedie_estimate)

        sigma_t_sq_val = sigma_t_sq(t)
        sigma_t_sq_val = broadcast_to_shape(sigma_t_sq_val, x.shape)

        if drift_estimator == "sne":
            steering_drift = steering_drift_fn(
                x, t, base_drift, **model_kwargs
            )  # 1/2\sigma_t^2 * steering
            steering_drift = 1 / (0.5 * sigma_t_sq_val) * steering_drift  # steering
        elif drift_estimator == "dps":
            steering_drift = steering_drift_fn(x, t, drift_ret["v"], **model_kwargs)
        else:
            steering_drift = steering_drift_fn(x, t, **model_kwargs)

        # detach both drifts
        base_drift = base_drift.detach()
        steering_drift = steering_drift.detach()

        # scale
        if type == "ode":
            steering_drift_scaled = 0.5 * sigma_t_sq_val * steering_drift
        elif type == "sde":
            steering_drift_scaled = sigma_t_sq_val * steering_drift

        # rescale
        if renorm_gradient:
            norm_base = torch.norm(
                base_drift.view(base_drift.shape[0], -1), dim=1, keepdim=True
            )  # [B, 1]
            norm_steering = torch.norm(
                steering_drift_scaled.view(steering_drift_scaled.shape[0], -1),
                dim=1,
                keepdim=True,
            )  # [B, 1]
            norm_base = broadcast_to_shape(norm_base, steering_drift_scaled.shape)
            norm_steering = broadcast_to_shape(
                norm_steering, steering_drift_scaled.shape
            )
            scale_factor = renorm_scale * norm_base / (1e-8 + norm_steering)

            if renorm_type == "clip":
                scale_factor = torch.clamp(scale_factor, max=1.0)
            elif renorm_type == "rescale":
                scale_factor = scale_factor
            else:
                raise ValueError(f"Unknown renorm type: {renorm_type}")

            steering_drift_scaled = steering_drift_scaled * scale_factor

        drift = base_drift + steering_drift_scaled * guidance_scale

        return drift, {
            "drift": drift,
            "uncond_drift": base_drift,
            "steering_drift": steering_drift,
            "steering_drift_scaled": steering_drift_scaled,
            "sigma_t_sq": sigma_t_sq_val,
            "tweedie_estimate": tweedie_estimate,
        }

    return drift_fn


def euler_maruyama_sampler(
    x0_start, drift_fn, t_start=0.01, n_steps=100, **model_kwargs
):
    N = x0_start.shape[0]
    x = x0_start
    t_steps = torch.linspace(t_start, 1.0, n_steps + 1, device=x.device)

    for i in tqdm(range(n_steps), desc="SDE sampling"):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        dt = t_next - t_cur
        t_batched = torch.full((N,), t_cur, device=x.device)
        drift, _ = drift_fn(x, t_batched, **model_kwargs)

        diffusion = torch.sqrt(sigma_t_sq(t_batched))
        diffusion = broadcast_to_shape(diffusion, x.shape)
        x = x + drift * dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)
    return x


def euler_sampler(x0_start, drift_fn, t_start=0.01, n_steps=100, **model_kwargs):
    N = x0_start.shape[0]
    x = x0_start
    t_steps = torch.linspace(t_start, 1.0, n_steps + 1, device=x.device)
    norm_base, norm_steering = [], []

    for i in tqdm(range(n_steps), desc="ODE sampling"):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        dt = t_next - t_cur
        t_batched = torch.full((N,), t_cur, device=x.device)
        drift, ret = drift_fn(x, t_batched, **model_kwargs)
        x = x + drift * dt

        # for logging
        base_drift = ret["uncond_drift"]
        steering_drift = ret["steering_drift_scaled"]
        norm_base.append(torch.mean(torch.norm(base_drift.view(N, -1), dim=1)).item())
        norm_steering.append(
            torch.mean(torch.norm(steering_drift.view(N, -1), dim=1)).item()
        )

    return x


def euler_sampler(x0_start, drift_fn, t_start=0.01, n_steps=100, **model_kwargs):
    N = x0_start.shape[0]
    x = x0_start
    t_steps = torch.linspace(t_start, 1.0, n_steps + 1, device=x.device)
    norm_base, norm_steering = [], []

    for i in tqdm(range(n_steps), desc="ODE sampling"):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        dt = t_next - t_cur
        t_batched = torch.full((N,), t_cur, device=x.device)
        drift, ret = drift_fn(x, t_batched, **model_kwargs)
        x = x + drift * dt

        # for logging
        base_drift = ret["uncond_drift"]
        steering_drift = ret["steering_drift_scaled"]
        norm_base.append(torch.mean(torch.norm(base_drift.view(N, -1), dim=1)).item())
        norm_steering.append(
            torch.mean(torch.norm(steering_drift.view(N, -1), dim=1)).item()
        )
    return x


def resample_for_tree_search(x1_samples, xt_samples, rewards, type="max"):
    B, K, mc_samples, C, H, W = x1_samples.shape
    # rewards: [B, K, mc_samples]
    if type == "max_per_particle":
        max_indices = torch.argmax(rewards, dim=2)  # [B, K]
        x1_samples = x1_samples[
            torch.arange(B).unsqueeze(1), torch.arange(K).unsqueeze(0), max_indices
        ]  # [B, K, C, H, W]
        xt_samples = xt_samples[
            torch.arange(B).unsqueeze(1), torch.arange(K).unsqueeze(0), max_indices
        ]  # [B, K, C, H, W]
    elif type == "max_overall":  # make for each of the element in the batch
        max_indices = torch.argmax(rewards.view(B, -1), dim=1)  # [B,]
        k_index = max_indices // mc_samples  # [B,]
        mc_index = max_indices % mc_samples  # [B,]
        x1_samples = x1_samples[torch.arange(B), k_index, mc_index]  # [B, C, H, W]
        xt_samples = xt_samples[torch.arange(B), k_index, mc_index]  # [B, C, H, W]
        x1_samples = x1_samples.unsqueeze(1).repeat(1, K, 1, 1, 1)  # [B, K, C, H, W]
        xt_samples = xt_samples.unsqueeze(1).repeat(1, K, 1, 1, 1)  # [B, K, C, H, W]
    else:
        raise ValueError(f"Unknown resampling type: {type}")
    return xt_samples, x1_samples


@torch.no_grad()
def euler_sampler_tree_search(
    x0_start,
    model,
    reward_fn,
    inverse_scaler,
    mc_samples_per_particle,
    t_start=0.01,
    n_steps=100,
    step_power=1.0,
    resampling_type="max_per_particle",
    transition_type="stochastic",  #
    **model_kwargs,
):
    B, K, *_ = x0_start.shape
    N = B * K
    x = x0_start.view(N, *x0_start.shape[2:])

    t_steps = torch.linspace(t_start, 1.0, n_steps + 1, device=x.device)
    t_steps = t_steps**step_power

    norm_base, norm_steering = [], []

    for i in tqdm(range(n_steps), desc="ODE sampling"):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        t_cur_batched = torch.full((N,), t_cur, device=x.device)
        t_next_batched = torch.full((N,), t_next, device=x.device)
        ret = batched_reward_estimation(
            model,
            x,
            t_cur_batched,
            reward_fn,
            inverse_scaler,
            mc_samples=mc_samples_per_particle,
            **model_kwargs,
        )
        xt_samples = ret["xt"]  # [N, mc_samples, C, H, W]
        x1_samples = ret["x1_samples"]  # [N, mc_samples, C, H, W]
        rewards = ret["reward"]  # [N, mc_samples]
        # reshape
        xt_samples = xt_samples.view(
            B, K, mc_samples_per_particle, *xt_samples.shape[2:]
        )  # [B, K, mc_samples, C, H, W]
        x1_samples = x1_samples.view(
            B, K, mc_samples_per_particle, *x1_samples.shape[2:]
        )  # [B, K, mc_samples, C, H, W]
        rewards = rewards.view(B, K, mc_samples_per_particle)  # [B, K, mc_samples]

        # [N, C, H, W]
        xt_samples, x1_samples = resample_for_tree_search(
            x1_samples, xt_samples, rewards, type=resampling_type
        )

        xt_samples = xt_samples.view(N, *xt_samples.shape[2:])  # [N, C, H, W]
        x1_samples = x1_samples.view(N, *x1_samples.shape[2:])  # [N, C, H, W]

        if transition_type == "stochastic":
            x = broadcast_to_shape(t_next, x1_samples) * x1_samples + (
                1 - broadcast_to_shape(t_next, x1_samples)
            ) * torch.randn_like(x1_samples)
        elif transition_type == "deterministic":
            eps = (
                xt_samples - broadcast_to_shape(t_cur, x1_samples) * x1_samples
            ) / broadcast_to_shape(1 - t_cur, x1_samples)
            x = (
                broadcast_to_shape(t_next, x1_samples) * x1_samples
                + (1 - broadcast_to_shape(t_next, x1_samples)) * eps
            )
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")

    return x.view(B, K, *x.shape[1:])  # [B, K, C, H, W]

