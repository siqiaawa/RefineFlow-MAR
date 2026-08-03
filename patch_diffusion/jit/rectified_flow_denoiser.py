"""Rectified-flow training and sampling wrapper for the JiT model."""

import torch
import torch.nn as nn

from .model_jit_rf import JiT_models


class RectifiedFlowDenoiser(nn.Module):
    """Wrap JiT with the x-prediction rectified-flow objective."""

    def __init__(self, args):
        super().__init__()
        self.net = JiT_models[args.model_name](
            input_size=args.image_size,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            attn_drop=args.attn_drop,
            proj_drop=args.proj_drop,
        )
        self.img_size = args.image_size
        self.in_channels = args.in_channels
        self.num_classes = args.num_classes

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg_scale
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        """Randomly replace class labels with the null class token."""
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def sample_t(self, n: int, device=None):
        """Sample timesteps from sigmoid(N(P_mean, P_std))."""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels=None):
        """Compute the mean squared x-prediction loss for a training batch."""
        if labels is not None:
            labels = self.drop_labels(labels) if self.training else labels
        else:
            labels = torch.full(
                (x.size(0),), self.num_classes, device=x.device, dtype=torch.long
            )

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        noise = torch.randn_like(x) * self.noise_scale
        noisy = t * x + (1.0 - t) * noise
        x_pred = self.net(noisy, t.flatten(), labels)
        return ((x - x_pred) ** 2).mean(dim=(1, 2, 3)).mean()

    @torch.no_grad()
    def generate(self, labels):
        """Generate samples with Euler or Heun x-prediction integration."""
        device = labels.device
        batch_size = labels.size(0)
        z = self.noise_scale * torch.randn(
            batch_size, self.in_channels, self.img_size, self.img_size, device=device
        )
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Sampling method {self.method} is not implemented")

        for index in range(self.steps):
            z = stepper(z, timesteps[index], timesteps[index + 1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        """Predict x0 and apply classifier-free guidance when requested."""
        t_batch = t.expand(z.size(0)) if t.dim() == 0 else t.flatten()
        x_cond = self.net(z, t_batch, labels)
        x_uncond = self.net(z, t_batch, torch.full_like(labels, self.num_classes))

        low, high = self.cfg_interval
        t_scalar = t.item() if t.dim() == 0 else t.flatten()[0].item()
        use_cfg = low <= t_scalar < high
        cfg = self.cfg_scale if use_cfg else 1.0
        return x_uncond + cfg * (x_cond - x_uncond)

    @staticmethod
    def _xpred_step(z, t, t_next, x_pred):
        """Take a numerically stable RF step without dividing by 1 - t."""
        if t_next >= 1.0:
            return x_pred
        ratio = (1.0 - t_next) / (1.0 - t)
        return ratio * z + (1.0 - ratio) * x_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        return self._xpred_step(z, t, t_next, self._forward_sample(z, t, labels))

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        x1 = self._forward_sample(z, t, labels)
        z_euler = self._xpred_step(z, t, t_next, x1)
        if t_next >= 1.0:
            return x1
        x2 = self._forward_sample(z_euler, t_next, labels)
        return self._xpred_step(z, t, t_next, 0.5 * (x1 + x2))

    @torch.no_grad()
    def update_ema(self):
        """Update the two exponential moving averages used for inference."""
        if self.ema_params1 is None:
            source_params = list(self.parameters())
            self.ema_params1 = [parameter.clone().detach() for parameter in source_params]
            self.ema_params2 = [parameter.clone().detach() for parameter in source_params]
            return

        source_params = list(self.parameters())
        for target, source in zip(self.ema_params1, source_params):
            target.detach().mul_(self.ema_decay1).add_(source, alpha=1.0 - self.ema_decay1)
        for target, source in zip(self.ema_params2, source_params):
            target.detach().mul_(self.ema_decay2).add_(source, alpha=1.0 - self.ema_decay2)
