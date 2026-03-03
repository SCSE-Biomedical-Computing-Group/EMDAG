import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emdag.modules.common.layers import clampped_one_hot
from emdag.modules.common.so3 import ApproxAngularDistribution, random_normal_so3, so3vec_to_rotation, rotation_to_so3vec

EPS = 1e-8


class VarianceSchedule(nn.Module):
    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = int(num_steps)
        t = torch.arange(0, T + 1, dtype=torch.float)

        f_t = torch.cos((np.pi / 2) * ((t / T) + s) / (1 + s)) ** 2
        alpha_bars = f_t / (f_t[0] + EPS)

        betas = 1 - (alpha_bars[1:] / (alpha_bars[:-1] + EPS))
        betas = torch.cat([torch.zeros([1]), betas], dim=0).clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i] + EPS)) * betas[i]
        sigmas = torch.sqrt(sigmas.clamp(min=0.0))

        self.register_buffer("betas", betas)           # (T+1,)
        self.register_buffer("alpha_bars", alpha_bars) # (T+1,)
        self.register_buffer("alphas", 1 - betas)      # (T+1,)
        self.register_buffer("sigmas", sigmas)         # (T+1,)

    def c1(self, t_long: torch.Tensor) -> torch.Tensor:
        """
        Convenience: sqrt(1 - alpha_bar_t), shape (N,1,1) for broadcasting.
        """
        a = self.alpha_bars[t_long].clamp(min=EPS, max=1.0)
        return torch.sqrt(1.0 - a).view(-1, 1, 1).clamp(min=EPS)


class PositionTransition(nn.Module):
    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t):
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_rand = torch.randn_like(p_0)
        p_noisy = c0 * p_0 + c1 * e_rand
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_0), p_noisy, p_0)
        return p_noisy, e_rand

    def denoise(self, p_t, eps_p, mask_generate, t):
        alpha = self.var_sched.alphas[t].clamp_min(self.var_sched.alphas[-2])
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)

        c0 = (1.0 / torch.sqrt(alpha + 1e-8)).view(-1, 1, 1)
        c1 = ((1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8)).view(-1, 1, 1)

        z = torch.where(
            (t > 1)[:, None, None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)
        return p_next


class RotationTransition(nn.Module):
    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        c1 = torch.sqrt(1 - self.var_sched.alpha_bars)  # (T+1,)
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)

        self.register_buffer("_dummy", torch.empty([0]))

    def add_noise(self, v_0, mask_generate, t):
        N, L = mask_generate.size()
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)

        e_scaled = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_fwd, device=self._dummy.device)
        E_scaled = so3vec_to_rotation(e_scaled)
        R0_scaled = so3vec_to_rotation(c0 * v_0)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)
        return v_noisy, e_scaled

    def denoise(self, v_t, v_next, mask_generate, t):
        N, L = mask_generate.size()
        e = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device)
        e = torch.where((t > 1)[:, None, None].expand(N, L, 3), e, torch.zeros_like(e))
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[..., None].expand_as(v_next), v_next, v_t)
        return v_next


class AminoacidCategoricalTransition(nn.Module):
    """
    Discrete forward with uniform mixing; posterior is closed-form (used for KL and reverse).
    """
    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)
        self.num_steps = int(num_steps)

    @staticmethod
    def _sample(c):
        """
        Args:
            c:    (N, L, K).
        Returns:
            x:    (N, L) Long
        """
        N, L, K = c.size()
        c = c.view(N * L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
        return x

    def add_noise(self, x_0, mask_generate, t):
        """
        Args:
            x_0:    (N, L) Long true labels in [0..K-1]
            mask_generate: (N,L) bool
            t:      (N,) long in [1..T]
        Returns:
            c_t:    (N,L,K) probabilities after corruption
            x_t:    (N,L) sampled labels
        """
        N, L = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float()  # (N,L,K)

        a = self.var_sched.alpha_bars[t][:, None, None]  # (N,1,1)
        c_noisy = (a * c_0) + ((1 - a) / K)
        c_t = torch.where(mask_generate[..., None].expand(N, L, K), c_noisy, c_0)
        x_t = self._sample(c_t)
        return c_t, x_t

    def posterior(self, x_t, x0_or_probs, t):
        """
        q(x0 | x_t) for the same uniform-mixing corruption.
        Args:
            x_t:            (N,L) Long or (N,L,K) probs
            x0_or_probs:    (N,L) Long or (N,L,K) probs (true one-hot or predicted c0)
            t:              (N,) long
        Returns:
            theta: (N,L,K)
        """
        K = self.num_classes

        # x_t -> onehot
        if x_t.dim() == 3:
            c_t = x_t
        else:
            c_t = F.one_hot(x_t.clamp(min=0), num_classes=K).float()

        # x0_or_probs -> probs
        if x0_or_probs.dim() == 3:
            c0 = x0_or_probs
            c0 = c0 / (c0.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            c0 = clampped_one_hot(x0_or_probs, num_classes=K).float()

        a = self.var_sched.alpha_bars[t][:, None, None]
        dot = (c0 * c_t).sum(dim=-1, keepdim=True)  # c0(j)
        theta = ((1 - a) / K) * c0 + a * dot * c_t
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-12)
        return theta

    def denoise(self, x_t, c0_pred, mask_generate, t):
        """
        Reverse categorical step using posterior with predicted c0.
        Args:
            x_t:        (N,L) Long labels at time t
            c0_pred:    (N,L,K) probs for x0
            mask_generate: (N,L)
            t:          (N,) long time index in [1..T]
        Returns:
            post:   (N,L,K) posterior
            x_prev: (N,L) sample
        """
        K = self.num_classes
        c0_pred = (c0_pred + 1e-12) / (c0_pred.sum(dim=-1, keepdim=True) + 1e-12)
        c_t = F.one_hot(x_t.clamp(min=0), num_classes=K).float()

        a = self.var_sched.alpha_bars[t][:, None, None]
        dot = (c0_pred * c_t).sum(dim=-1, keepdim=True)
        theta = ((1 - a) / K) * c0_pred + a * dot * c_t

        theta = torch.where(mask_generate[..., None], theta, c_t)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-12)

        x_prev = torch.multinomial(theta.view(-1, K), 1).view(x_t.shape[0], x_t.shape[1])
        return theta, x_prev

