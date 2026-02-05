import torch


def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))


def l2_loss(
    pred,
    target,
    weighting,
    stop_gradient=False,
):
    """Computes the mean squared L2 loss."""
    if stop_gradient:
        actual_target = torch.detach(target)
    else:
        actual_target = target
    delta_sq = (pred - actual_target) ** 2
    # sum over data dimensions
    delta_sq = torch.sum(delta_sq, dim=list(range(1, len(delta_sq.shape))))
    weighting = broadcast_to_shape(weighting, delta_sq.shape)
    weighted_delta_sq = (1 / weighting.exp()) * delta_sq + weighting
    # mean over batch
    return torch.mean(weighted_delta_sq), torch.mean(delta_sq)


def log_lv_loss(pred, target, weighting, stop_gradient=False):
    """Computes the mean squared L2 loss."""
    if stop_gradient:
        actual_target = torch.detach(target)
    else:
        actual_target = target
    delta_sq = (pred - actual_target) ** 2
    mean_loss = torch.mean(delta_sq, dim=list(range(1, len(pred.shape))))
    # Reshape mean_loss to match weighting for broadcasting: (B,) -> (B, 1, 1, 1)
    mean_loss = broadcast_to_shape(mean_loss, weighting.shape)
    log_loss = torch.log((1 / weighting.exp()) * mean_loss + 1.0) + 0.5 * weighting
    return torch.mean(log_loss), torch.mean(mean_loss)


def adaptive_loss(pred, target, weighting, p, c, stop_gradient=False):
    """Computes the adaptively weighted squared L2 loss.
    Loss = w * ||pred - target||^2, where w = 1 / (||pred - target||^2 + c)^p.
    """
    if stop_gradient:
        actual_target = torch.detach(target)
    else:
        actual_target = target

    delta_sq = (pred - actual_target) ** 2
    # sum over data dimensions
    delta_sq = torch.sum(delta_sq, dim=tuple(range(1, len(delta_sq.shape))))
    weight = 1.0 / (delta_sq + c) ** p
    weight = torch.detach(weight)
    weight = broadcast_to_shape(weight, delta_sq.shape)
    delta_sq = delta_sq * weight
    weighted_delta_sq = delta_sq / weighting.exp() + weighting
    # mean over batch
    return torch.mean(weighted_delta_sq), torch.mean(delta_sq).detach()


def compute_loss(
    pred,
    target,
    weighting,
    loss_type,
    adaptive_p=None,
    adaptive_c=None,
    stop_gradient=False,
):
    if loss_type == "l2":
        return l2_loss(pred, target, weighting, stop_gradient=stop_gradient)
    elif loss_type == "lv":
        return log_lv_loss(pred, target, weighting, stop_gradient=stop_gradient)
    elif loss_type == "adaptive":
        return adaptive_loss(
            pred, target, weighting, adaptive_p, adaptive_c, stop_gradient=stop_gradient
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
