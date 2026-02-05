import torch

from mfm.losses.utils import compute_loss


def sample_t_cond(N, step, cfg):
    """Samples conditioning time t_cond with an annealing schedule."""
    n_warmup_steps = cfg.trainer.t_cond_warmup_steps
    t_max = cfg.SI.t_max

    if step < n_warmup_steps:
        # Phase 1: Learning the standard flow map.
        return torch.zeros(N)
    else:
        # Phase 2: Train on the full range of noise levels.
        probs = torch.full((N,), 1.0 - cfg.trainer.t_cond_0_rate)
        return (
            torch.rand(N) ** cfg.trainer.t_cond_power * t_max * torch.bernoulli(probs)
        )


def sample_s_u(N, step, cfg):
    n_warmup_steps = cfg.trainer.num_warmup_steps
    anneal_end_step = cfg.trainer.anneal_end_step

    t_batch = torch.rand(N, 2) * cfg.SI.t_max  # Shape: [B, 2]
    t1, t2 = t_batch[:, 0], t_batch[:, 1]

    t_min = torch.min(t1, t2)
    t_max = torch.max(t1, t2)

    mid = (t_min + t_max) / 2
    dist = t_max - t_min

    def warmup_phase():
        # Phase 1: Learning the diagonal.
        return t1, t1

    def anneal_phase():
        # Phase 2: Expanding the jump.
        anneal_duration = anneal_end_step - n_warmup_steps
        progress = (step - n_warmup_steps) / max(anneal_duration, 1)
        max_step_size = torch.clamp(torch.tensor(progress), min=0, max=1)
        s = mid - max_step_size * dist / 2
        t = mid + max_step_size * dist / 2
        return s, t

    def final_phase():
        # Phase 3: All jump sizes.
        return t_min, t_max

    if step < n_warmup_steps:
        return warmup_phase()
    elif step < anneal_end_step:
        return anneal_phase()
    else:
        return final_phase()


def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))


# learn the standard flow for dropped classes
def generate_cfg_values(ws, base_prob, mask):
    device = mask.device
    N = mask.shape[0]
    ws = torch.tensor(ws, device=device)
    cfg_scale = ws[torch.randint(len(ws), (N,), device=device)]
    cfg_mask = torch.bernoulli(torch.full((N,), base_prob, device=device)).bool()
    cfg_scales = torch.where(
        cfg_mask, torch.ones_like(cfg_scale, device=device), cfg_scale
    )
    cfg_scales = torch.where(
        mask, torch.ones_like(cfg_scales, device=device), cfg_scales
    )
    return cfg_scales


def model_guidance_target(
    model, s, t, x, t_cond, x_cond, class_labels, cfg_scales, label_dim
):
    assert torch.equal(s, t), "implemented for velocity only!"
    device = s.device
    s_2 = torch.cat([s, s], dim=0)
    t_2 = torch.cat([t, t], dim=0)
    x_2 = torch.cat([x, x], dim=0)
    t_cond_2 = torch.cat([t_cond, t_cond], dim=0)
    x_cond_2 = torch.cat([x_cond, x_cond], dim=0)

    null_labels = torch.full((x.shape[0],), label_dim, dtype=torch.long, device=device)
    labels = torch.cat([null_labels, class_labels], dim=0)
    cfg_scales = torch.cat([torch.ones_like(cfg_scales), cfg_scales], dim=0)

    v = model.v(
        s_2, t_2, x_2, t_cond_2, x_cond_2, class_labels=labels, cfg_scale=cfg_scales
    )
    v_uncond, v_mg = v.chunk(2, dim=0)
    return v_uncond, v_mg


@torch.no_grad()
def extract_posterior_velocity(
    s,
    Is,
    xt_cond,
    t_cond,
    labels,
    cfg_scales,
    teacher_model,
    eps=1e-6,
    omega=0.6,
    checkpoint_type="dmf",
):
    device = s.device
    N = s.shape[0]
    s = broadcast_to_shape(s, xt_cond.shape)
    t = broadcast_to_shape(t_cond, xt_cond.shape)
    one_minus_s = 1 - s  # 1

    denom = (t**2 * one_minus_s**2 + (1 - t) ** 2 * s**2).clamp_min(eps)  # t**2
    P_norm = (1 - t) ** 2 / denom  # ((1-t)**2 / t**2)
    sqrt_P_norm = torch.sqrt(P_norm)  # (1-t)/t

    t_star = 1 / (1 + one_minus_s * sqrt_P_norm)  # t

    coeff_cond = t_star * one_minus_s**2 * t / denom  # 1
    coeff_Is = t_star * s * P_norm  # 0
    x_star = coeff_cond * xt_cond + coeff_Is * Is  # x_t_cond

    with torch.no_grad():
        # rearrange v_mg = 1/(1-\omega)*(v_c - \omega*v_uncond)
        if labels is None:
            v_star = teacher_model.v(
                t_star.view(N),
                t_star.view(N),
                x_star,
                torch.zeros(N, device=device),
                torch.zeros_like(x_star),
            )
        else:
            if checkpoint_type == "dmf":
                v_star_uncond, v_star_mg = model_guidance_target(
                    teacher_model,
                    t_star.view(N),
                    t_star.view(N),
                    x_star,
                    torch.zeros(N, device=device),
                    torch.zeros_like(x_star),
                    class_labels=labels,
                    cfg_scales=torch.ones_like(t_star).view(N),
                    label_dim=1000,
                )
                v_star_cond = v_star_mg * (1 - omega) + omega * v_star_uncond
                v_star = v_star_uncond + broadcast_to_shape(
                    cfg_scales, v_star_uncond.shape
                ) * (
                    v_star_cond - v_star_uncond
                )  # b_t(x_t)
            elif checkpoint_type == "sit":
                v_star_uncond, v_star_cond = model_guidance_target(
                    teacher_model,
                    t_star.view(N),
                    t_star.view(N),
                    x_star,
                    torch.zeros(N, device=device),
                    torch.zeros_like(x_star),
                    class_labels=labels,
                    cfg_scales=torch.ones_like(t_star).view(N),
                    label_dim=1000,
                )
                v_star = v_star_uncond + broadcast_to_shape(
                    cfg_scales, v_star_uncond.shape
                ) * (v_star_cond - v_star_uncond)

        term2 = t_star * sqrt_P_norm * v_star  # (1-t)b_t(x_t)
        diff_div_x = (
            (1 - t) ** 2 * (1 + s) - t**2 * one_minus_s
        ) / denom  # (1-2t)/t**2
        B_minus_1_div_x = (diff_div_x - (P_norm + sqrt_P_norm)) / (
            1 + one_minus_s * sqrt_P_norm
        )  # -1
        A_div_x = t_star * one_minus_s * t / denom  # 1
        # x_t - I_s
        term1 = A_div_x * xt_cond + B_minus_1_div_x * Is
        # x_t - I_s + (1-t)b_t(x_t) -> x_t + (1-t)b_t(x_t) -> E[x_1|x_t]
        dIsds_distill = term1 + term2

    return dIsds_distill


def get_consistency_loss_fn(cfg, SI):
    def loss_fn(model, weighting_model, x1, labels, step, teacher_model=None):
        device = x1.device
        # ---- 0. Drop classes ----
        if cfg.model.label_dim > 0 and cfg.trainer.class_dropout_prob > 0:
            prob = cfg.trainer.class_dropout_prob
            dropout_mask = torch.bernoulli(
                torch.full(labels.shape, prob, device=device)
            ).bool()
            null_class_token = torch.tensor([cfg.model.label_dim], device=device)
            labels = torch.where(
                dropout_mask, null_class_token.expand_as(labels), labels
            )

        # --- 1. Generate Conditioning Variables ---
        N = x1.shape[0]  # batch size
        t_cond = sample_t_cond(N, step, cfg)  # Shape: [B,]
        t_cond = t_cond.to(device)

        alpha_t_cond, beta_t_cond = SI.get_coefficients(t_cond)  # Shape: [B,]
        alpha_t_cond, beta_t_cond = broadcast_to_shape(
            alpha_t_cond, x1.shape
        ), broadcast_to_shape(beta_t_cond, x1.shape)

        noise_cond = torch.randn_like(x1, device=x1.device)
        xt_cond = alpha_t_cond * noise_cond + beta_t_cond * x1

        # --- 2. Flow Matching Loss (on the diagonal) ---
        x0 = torch.randn_like(x1)
        s_uniform = torch.rand(
            N,
        )
        s_uniform = s_uniform.to(device)
        expanded_s_uniform = broadcast_to_shape(s_uniform, x1.shape)
        Is = (1 - expanded_s_uniform) * x0 + expanded_s_uniform * x1
        # Data FM
        if cfg.loss.data_fm:
            fm_target = x1 - x0

            if cfg.loss.model_guidance:
                assert (
                    len(cfg.model.model_guidance_class_ws) > 0
                ), "Model guidance class weights must be provided."
                cfg_scales = generate_cfg_values(
                    cfg.model.model_guidance_class_ws,
                    cfg.loss.model_guidance_base_prob,
                    mask=dropout_mask,
                )
                mg_scales = 1 - (1 / cfg_scales)
                with torch.no_grad():
                    v_uncond, v_mg = model_guidance_target(
                        model,
                        s_uniform,
                        s_uniform,
                        Is,
                        t_cond,
                        xt_cond,
                        class_labels=labels,
                        cfg_scales=cfg_scales,
                        label_dim=cfg.model.label_dim,
                    )
                fm_target = fm_target + broadcast_to_shape(
                    mg_scales, fm_target.shape
                ) * (v_mg - v_uncond)
            else:
                cfg_scales = torch.ones(N, device=device)
            fm_pred = model.v(
                s_uniform,
                s_uniform,
                Is,
                t_cond,
                xt_cond,
                class_labels=labels,
                cfg_scale=cfg_scales,
            )

            if cfg.model.learn_loss_weighting:
                fm_loss_weighting = weighting_model(s_uniform, t_cond)
            else:
                fm_loss_weighting = torch.zeros_like(t_cond)

            fm_loss, fm_loss_unweighted = compute_loss(
                fm_pred,
                fm_target,
                fm_loss_weighting,
                cfg.loss.fm_loss_type,
                adaptive_p=cfg.loss.get("fm_adaptive_loss_p"),
                adaptive_c=cfg.loss.get("fm_adaptive_loss_c"),
                stop_gradient=True,
            )

            # for logging
            with torch.no_grad():
                fm_loss_l2 = (fm_pred - fm_target) ** 2
                if cfg.loss.model_guidance:
                    cfg_mask = (cfg_scales == 1).bool()
                    fm_loss_l2_base = (
                        fm_loss_l2[cfg_mask]
                        .sum(dim=tuple(range(1, fm_loss_l2.ndim)))
                        .mean()
                    )
                    fm_loss_l2_mg = (
                        fm_loss_l2[~cfg_mask]
                        .sum(dim=tuple(range(1, fm_loss_l2.ndim)))
                        .mean()
                    )
                else:
                    fm_loss_l2_base = fm_loss_l2.sum(
                        dim=tuple(range(1, fm_loss_l2.ndim))
                    ).mean()
                    fm_loss_l2_mg = torch.tensor(0.0, device=device)
        else:
            fm_loss = torch.tensor(0.0, device=device)
            fm_loss_unweighted = torch.tensor(0.0, device=device)
            fm_loss_l2_base = torch.tensor(0.0, device=device)
            fm_loss_l2_mg = torch.tensor(0.0, device=device)

        # Distilled FM
        if cfg.loss.distill_fm:
            if cfg.loss.model_guidance:
                assert (
                    len(cfg.model.model_guidance_class_ws) > 0
                ), "Model guidance class weights must be provided."
                cfg_scales = generate_cfg_values(
                    cfg.model.model_guidance_class_ws,
                    cfg.loss.model_guidance_base_prob,
                    mask=dropout_mask,
                )

            # Predict
            if not cfg.loss.data_fm:
                fm_pred = model.v(
                    s_uniform,
                    s_uniform,
                    Is,
                    t_cond,
                    xt_cond,
                    class_labels=labels,
                    cfg_scale=cfg_scales,
                )

            # Compute distilled target
            teacher_model_type = cfg.model.init
            dIsds_distill = extract_posterior_velocity(
                s_uniform,
                Is,
                xt_cond,
                t_cond,
                labels,
                cfg_scales,
                teacher_model,
                eps=1e-6,
                omega=0.6,
                checkpoint_type=teacher_model_type,
            )

            # Calculate loss
            distill_loss_weighting = torch.zeros_like(t_cond)
            distill_fm_loss, distill_fm_loss_unweighted = compute_loss(
                fm_pred,
                dIsds_distill,
                distill_loss_weighting,
                cfg.loss.distill_fm_loss_type,
                adaptive_p=cfg.loss.get("fm_adaptive_loss_p"),
                adaptive_c=cfg.loss.get("fm_adaptive_loss_c"),
            )
            with torch.no_grad():
                distill_fm_loss_l2 = (fm_pred - dIsds_distill) ** 2
                distill_fm_loss_l2 = distill_fm_loss_l2.sum(
                    dim=tuple(range(1, distill_fm_loss_l2.ndim))
                )
                distill_fm_loss_l2 = distill_fm_loss_l2.mean()
        else:
            distill_fm_loss = torch.tensor(0.0, device=device)
            distill_fm_loss_unweighted = torch.tensor(0.0, device=device)
            distill_fm_loss_l2 = torch.tensor(0.0, device=device)

        # --- 3. Distillation Loss (on the off-diagonal s<u) ---
        if step > cfg.trainer.num_warmup_steps:
            # sample noise and compute velocity
            x0 = torch.randn_like(x1)
            fm_target = x1 - x0
            # sample s,u,interpolant
            s, u = sample_s_u(N, step, cfg)  # [B,]
            s, u = s.to(device), u.to(device)
            expanded_s = broadcast_to_shape(s, x1.shape)
            Is = (1 - expanded_s) * x0 + expanded_s * x1

            if cfg.loss.model_guidance:
                cfg_scales_distill = generate_cfg_values(
                    cfg.model.model_guidance_class_ws,
                    cfg.loss.model_guidance_base_prob,
                    mask=dropout_mask,
                )
            else:
                cfg_scales_distill = torch.ones(N, device=device)

            # Define average velocity functions
            vsu_fn = lambda s, u, x: model.v(
                s,
                u,
                x,
                t_cond,
                xt_cond,
                class_labels=labels,
                cfg_scale=cfg_scales_distill,
            )
            Xsu_fn = lambda s, u, x: model(
                s,
                u,
                x,
                t_cond,
                xt_cond,
                class_labels=labels,
                cfg_scale=cfg_scales_distill,
            )

            if cfg.loss.distillation_type == "lsd":
                primals = (s, u, Is)
                tangents = (
                    torch.zeros_like(s, device=device),
                    torch.ones_like(u, device=device),
                    torch.zeros_like(Is, device=device),
                )
                Xsu, dXdu = torch.func.jvp(Xsu_fn, primals, tangents)
                vuu = model.v(
                    u,
                    u,
                    Xsu,
                    t_cond,
                    xt_cond,
                    class_labels=labels,
                    cfg_scale=cfg_scales_distill,
                )

                distillation_student = vuu
                distillation_teacher = dXdu
            elif cfg.loss.distillation_type == "mf":  # vss from data
                if cfg.loss.model_guidance:
                    with torch.no_grad():
                        v_uncond, v_mg = model_guidance_target(
                            model,
                            s,
                            s,
                            Is,
                            t_cond,
                            xt_cond,
                            class_labels=labels,
                            cfg_scales=cfg_scales_distill,
                            label_dim=cfg.model.label_dim,
                        )
                    mg_scales_distill = 1 - (1 / cfg_scales_distill)
                    vss = fm_target + broadcast_to_shape(
                        mg_scales_distill, fm_target.shape
                    ) * (v_mg - v_uncond)
                else:
                    vss = fm_target
                primals = (s, u, Is)
                tangents = (
                    torch.ones_like(s, device=device),
                    torch.zeros_like(u, device=device),
                    vss,
                )
                vsu, jvp = torch.func.jvp(vsu_fn, primals, tangents)

                distillation_student = vsu
                distillation_teacher = vss + broadcast_to_shape(u - s, jvp.shape) * jvp
            elif cfg.loss.distillation_type == "esd_teacher":  # both from teacher
                teacher_model_type = cfg.model.init
                vss = extract_posterior_velocity(
                    s,
                    Is,
                    xt_cond,
                    t_cond,
                    labels,
                    cfg_scales_distill,
                    teacher_model,
                    eps=1e-6,
                    omega=0.6,
                    checkpoint_type=teacher_model_type,
                )

                primals = (s, u, Is)
                tangents = (
                    torch.ones_like(s, device=device),
                    torch.zeros_like(u, device=device),
                    vss,
                )
                vsu, jvp = torch.func.jvp(vsu_fn, primals, tangents)

                distillation_student = vsu
                distillation_teacher = vss + broadcast_to_shape(u - s, jvp.shape) * jvp
            elif cfg.loss.distillation_type == "psd":
                with torch.no_grad():
                    gamma = torch.rand_like(s, device=device)
                    w = s + gamma * (u - s)
                    vsw = model.v(
                        s,
                        w,
                        Is,
                        t_cond,
                        xt_cond,
                        class_labels=labels,
                        cfg_scale=cfg_scales_distill,
                    )
                    Xsw = model.X(s, w, Is, vsw)

                expanded_gamma = broadcast_to_shape(gamma, x1.shape)
                distillation_student = vsu_fn(s, u, Is)
                distillation_teacher = expanded_gamma * vsw + (
                    1 - expanded_gamma
                ) * vsu_fn(w, u, Xsw)
            else:
                raise ValueError(
                    f"Unknown distillation loss type: {cfg.loss.distillation_type}"
                )

            distill_loss_weighting = torch.zeros_like(t_cond)

            distillation_loss, distillation_loss_unweighted = compute_loss(
                distillation_student,
                distillation_teacher,
                distill_loss_weighting,
                cfg.loss.distillation_loss_type,
                adaptive_p=cfg.loss.get("distill_adaptive_loss_p"),
                adaptive_c=cfg.loss.get("distill_adaptive_loss_c"),
                stop_gradient=cfg.loss.distill_teacher_stop_grad,
            )

            with torch.no_grad():
                distillation_loss_l2 = (
                    distillation_student - distillation_teacher
                ) ** 2
                distillation_loss_l2 = distillation_loss_l2.sum(
                    dim=tuple(range(1, distillation_loss_l2.ndim))
                )
                distillation_loss_l2 = distillation_loss_l2.mean()
        else:
            distillation_loss = torch.tensor(0.0, device=device)
            distillation_loss_unweighted = torch.tensor(0.0, device=device)
            distillation_loss_l2 = torch.tensor(0.0, device=device)

        optimisation_losses = {
            "fm_loss": fm_loss,
            "distill_fm_loss": distill_fm_loss,
            "distillation_loss": distillation_loss,
        }

        logging_losses = {
            "fm_loss_unweighted": fm_loss_unweighted,
            "distill_fm_loss_unweighted": distill_fm_loss_unweighted,
            "distillation_loss_unweighted": distillation_loss_unweighted,
            "fm_loss_l2_base": fm_loss_l2_base,
            "fm_loss_l2_mg": fm_loss_l2_mg,
            "distill_fm_loss_l2": distill_fm_loss_l2,
            "distillation_loss_l2": distillation_loss_l2,
        }

        return optimisation_losses, logging_losses

    return loss_fn
