import torch
from timm.data import ImageNetInfo


def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))


def sigma_t_sq(t, cfg):
    if cfg.loss.get("sigma_t_sq_type", "constant") == "constant":
        return torch.ones_like(t) * cfg.loss.sigma_t_sq
    else:
        return 2 * torch.clamp((1 / (t + 1e-8) - 1), min=0, max=1000)


def get_finetuning_loss_fn(cfg, SI):
    class_description = ImageNetInfo()
    prompt_prefix = cfg.loss.get("prompt_prefix", "")

    def loss_fn(
        model, base_model, mfm, reward_fn, x1, labels, step, inverse_scaler_fn=None
    ):
        N = x1.shape[0]

        prompts = []

        if cfg.loss.get("prompt_type", "auto") == "auto":
            for label in labels:
                label = label.int()
                class_desc = class_description.index_to_description(label)
                prompts.append(f"{prompt_prefix}{class_desc}")
        else:
            prompts.append(cfg.loss.prompt)

        N = x1.shape[0]
        # sample time
        t = torch.rand(N, device=x1.device)
        noise = torch.randn_like(x1)

        t_broadcasted = broadcast_to_shape(t, x1.shape)
        # interpolant
        xt = t_broadcasted * x1 + (1 - t_broadcasted) * noise

        # sample x1 | xt
        xt = xt.detach().requires_grad_(True)  # Enable gradient tracking

        eps = torch.randn_like(xt)
        with torch.enable_grad():
            v_pred = mfm.v(
                torch.zeros_like(t, device=x1.device),
                torch.ones_like(t, device=x1.device),
                eps,
                t,
                xt,
                class_labels=labels,
                cfg_scale=torch.full_like(t, cfg.get("cfg_scale", 2.5)),
            )
            x1_samps = mfm.X(
                torch.zeros_like(t, device=x1.device),
                torch.ones_like(t, device=x1.device),
                eps,
                v_pred,
            )
            if inverse_scaler_fn is not None:
                x1_samps = inverse_scaler_fn(x1_samps)

            rewards = cfg.loss.get("_lambda", 1.0) * reward_fn.score(
                x1_samps, prompts
            )
            exp_r = torch.exp(rewards)
            grad_x = torch.autograd.grad(exp_r.sum(), inputs=xt)[0]

        # predict bt
        bt_pred = model(xt, t, labels)
        pred = bt_pred

        # form regression target
        bt_base = base_model(xt, t, labels)

        exp_r_broadcast = broadcast_to_shape(exp_r, bt_pred.shape)
        sq_sigma_t = sigma_t_sq(t, cfg)
        sq_sigma_t = broadcast_to_shape(sq_sigma_t, bt_pred.shape)

        # Calculate terms for logging
        term_base = bt_base
        term_shift = (exp_r_broadcast - 1) * (bt_pred - bt_base)
        term_grad = 0.5 * sq_sigma_t * grad_x

        target = term_base - term_shift + term_grad
        target = target.detach()

        # Calculate ratios of squared norms
        with torch.no_grad():
            norm_base = term_base.pow(2).sum()
            norm_shift = term_shift.pow(2).sum()
            norm_grad = term_grad.pow(2).sum()
            total_norm = norm_base + norm_shift + norm_grad

            ratio_base = norm_base / (total_norm + 1e-8)
            ratio_shift = norm_shift / (total_norm + 1e-8)
            ratio_grad = norm_grad / (total_norm + 1e-8)

            grad_x_norm = grad_x.norm()
            bt_base_norm = bt_base.norm()

            expected_reward = rewards.mean()
            avg_sigma_sq = sq_sigma_t.mean()

        loss = (pred - target).pow(2).mean()
        return {"loss": loss}, {
            "ratio_base": ratio_base,
            "ratio_shift": ratio_shift,
            "ratio_grad": ratio_grad,
            "grad_x_norm": grad_x_norm,
            "bt_base_norm": bt_base_norm,
            "expected_reward": expected_reward,
            "avg_sigma_sq": avg_sigma_sq,
        }
        
    return loss_fn
