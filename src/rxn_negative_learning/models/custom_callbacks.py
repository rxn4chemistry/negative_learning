from pytorch_lightning import Callback


class Grad2NormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        model.log("my_model/grad2_norm", gradient2_norm(model))


def gradient2_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


class MaxGradCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        model.log("my_model/max_grad", max_gradient(model))


def max_gradient(model):
    max_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.abs().max()
            max_norm = max(max_norm, param_norm)
    return max_norm
