from mfm.models.base_model import BaseModel


def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))


class SIModelWrapper(BaseModel):
    """A wrapper to handle SI-specific parametrizations for a model."""

    def __init__(self, model, SI, use_parametrization=False):
        super().__init__()
        self.model = model
        self.SI = SI
        self.use_parametrization = use_parametrization

    def v(self, s, u, x, t_cond, x_cond, **kwargs):
        """
        The wrapper's velocity method. It transforms inputs before calling the
        underlying model.
        """
        if self.use_parametrization:
            # Only for Linear
            result = self.model.v(s, u, x, t_cond, x_cond, **kwargs)

            if isinstance(result, tuple):
                v, *extra = result
            else:
                v, extra = result, []

            s_b = broadcast_to_shape(s, x.shape)
            t_cond_b = broadcast_to_shape(t_cond, x.shape)
            v = (1 - t_cond_b) * v + t_cond_b / (1 - s_b) * (x_cond - x)

            if extra:
                return (v, *extra)
            else:
                return v
        else:
            return self.model.v(s, u, x, t_cond, x_cond, **kwargs)

    def v_cfg(self, s, t, x, t_cond, x_cond, class_labels, cfg_scales, **kwargs):
        if self.use_parametrization:
            raise NotImplementedError
        else:
            return self.model.v_cfg(
                s, t, x, t_cond, x_cond, class_labels, cfg_scales, **kwargs
            )
