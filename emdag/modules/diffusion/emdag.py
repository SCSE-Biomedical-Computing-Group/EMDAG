"""Started from t = 1 instead"""
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from emdag.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from emdag.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from emdag.modules.encoders.ga import GAEncoder
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition

EPS = 1e-8


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3)
        R_true: (*, 3, 3)
    Returns:
        Per-matrix loss: (*,)
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3
    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)
    ones = torch.ones([ncol], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction="none")
    loss = loss.reshape(size + [3]).sum(dim=-1)
    return loss


class EpsilonNet(nn.Module):
    """
    Adds an energy head E_phi. Energy is conditioned on the same encoder features.

    Returns:
        v_next, R_next, eps_pos, c_denoised, E
    where:
        E: (N,) scalar energy per structure (sum over generated residues).
    """

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}, use_energy=True):
        super().__init__()
        self.use_energy = bool(use_energy)

        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, 3),
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, 3),
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim + 3, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, 20),
            nn.Softmax(dim=-1),
        )

        if self.use_energy:
            self.energy_net = nn.Sequential(
                nn.Linear(res_feat_dim + 3, res_feat_dim),
                nn.ReLU(),
                nn.Linear(res_feat_dim, res_feat_dim),
                nn.ReLU(),
                nn.Linear(res_feat_dim, 1),
            )

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res):
        """
        Args:
            v_t: (N,L,3)
            p_t: (N,L,3)
            s_t: (N,L)
            beta: (N,)  (time embedding feature)
        Returns:
            v_next: (N,L,3)
            R_next: (N,L,3,3)
            eps_pos: (N,L,3)  (global)
            c_denoised: (N,L,20)
            E: (N,) or None
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t)  # (N,L,3,3)

        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1))
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        # lightweight time embedding
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :]  # (N,1,3)
        in_feat = torch.cat([res_feat, t_embed.expand(N, L, 3)], dim=-1)

        # Position epsilon (local -> global)
        eps_crd = self.eps_crd_net(in_feat)              # (N,L,3) local
        eps_pos = apply_rotation_to_vector(R, eps_crd)   # (N,L,3) global
        eps_pos = eps_pos * mask_generate[:, :, None].to(eps_pos.dtype)

        # Rotation update
        eps_rot = self.eps_rot_net(in_feat)                      # (N,L,3)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot)          # (N,L,3,3)
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[:, :, None], v_next, v_t)

        # Sequence categorical distribution
        c_denoised = self.eps_seq_net(in_feat)  # (N,L,20)

        # Energy
        E = None
        if self.use_energy:
            e_res = self.energy_net(in_feat).squeeze(-1)  # (N,L)
            e_res = e_res * mask_generate.to(e_res.dtype)
            E = e_res.sum(dim=1)  # (N,)

        return v_next, R_next, eps_pos, c_denoised, E


class FullDPM(nn.Module):
    """
    - Energy head can exist (use_energy=True)
    - Energy Matching loss weight is auto-scaled each step using EMA stats
    """

    def __init__(
        self,
        res_feat_dim,
        pair_feat_dim,
        num_steps,
        eps_net_opt={},
        trans_rot_opt={},
        trans_pos_opt={},
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
        use_energy=True,
        # Auto Energy Matching knobs 
        em_enabled: bool = True,
        em_anchor: str = "pos",
        em_target_ratio: float = 0.25,
        em_ema_beta: float = 0.99,
        em_warmup_steps: int = 200,
        em_lambda_max: float = 5.0,
        em_use_pred_target: bool = False,
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, use_energy=use_energy, **eps_net_opt)
        self.num_steps = int(num_steps)
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer("position_mean", torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer("position_scale", torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer("_dummy", torch.empty([0]))

        self.use_energy = bool(use_energy)
        self.em_enabled = bool(em_enabled) and self.use_energy
        self.em_anchor = str(em_anchor)
        self.em_target_ratio = float(em_target_ratio)
        self.em_ema_beta = float(em_ema_beta)
        self.em_warmup_steps = int(em_warmup_steps)
        self.em_lambda_max = float(em_lambda_max)
        self.em_use_pred_target = bool(em_use_pred_target)

        self.register_buffer("em_step", torch.zeros([], dtype=torch.long))
        self.register_buffer("ema_anchor", torch.ones([], dtype=torch.float))
        self.register_buffer("ema_em", torch.ones([], dtype=torch.float))

    def _normalize_position(self, p):
        return (p - self.position_mean) / self.position_scale

    def _unnormalize_position(self, p_norm):
        return p_norm * self.position_scale + self.position_mean

    def _get_anchor_value(self, loss_dict: dict) -> torch.Tensor:
        if self.em_anchor in loss_dict:
            return loss_dict[self.em_anchor].detach()
        return loss_dict["pos"].detach()

    def _warmup_factor(self) -> torch.Tensor:
        if self.em_warmup_steps <= 0:
            return torch.ones_like(self.em_step, dtype=torch.float)
        step_f = self.em_step.float()
        return torch.clamp(step_f / float(self.em_warmup_steps), 0.0, 1.0)

    def forward(
        self,
        v_0, p_0, s_0,
        res_feat, pair_feat,
        mask_generate, mask_res,
        denoise_structure, denoise_sequence,
        t=None,
    ):
        """
        sample t in [1..T] (exclude 0) so EM target doesn't blow up from c1≈0.
        removed 2nd eps_net pass by making p_noisy a leaf requires_grad ONLY when EM is enabled,
        then compute grad_E from the SAME forward.
        """
        N, L = res_feat.shape[:2]
        device = self._dummy.device

        if t is None:
            t = torch.randint(1, self.num_steps + 1, (N,), dtype=torch.long, device=device)

        p_0n = self._normalize_position(p_0)

        if denoise_structure:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            p_noisy, eps_p = self.trans_pos.add_noise(p_0n, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0
            p_noisy = p_0n
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0

        beta = self.trans_pos.var_sched.betas[t]  # (N,)

        # single-pass EM that only require grad on p when needed
        em_active = (
            self.em_enabled
            and denoise_structure
            and torch.is_grad_enabled()
            and getattr(self.eps_net, "use_energy", False)
        )

        if em_active:
            # leaf tensor so autograd.grad wrt p works + enables higher-order grads to params via create_graph=True
            p_in = p_noisy.detach().clone().requires_grad_(True)
            v_in = v_noisy.detach()  # v grad not needed for EM here
            s_in = s_noisy.detach()
        else:
            p_in = p_noisy
            v_in = v_noisy
            s_in = s_noisy

        v_pred, R_pred, eps_p_pred, c_denoised, E = self.eps_net(
            v_in, p_in, s_in, res_feat, pair_feat, beta, mask_generate, mask_res
        )

        denom = mask_generate.sum().float() + 1e-8

        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0)
        loss_rot = (loss_rot * mask_generate).sum() / denom

        # Position loss (eps MSE)
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction="none").sum(dim=-1)
        loss_pos = (loss_pos * mask_generate).sum() / denom

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        post_pred = self.trans_seq.posterior(s_noisy, c_denoised, t)
        log_post_pred = torch.log(post_pred + 1e-8)
        kldiv = F.kl_div(input=log_post_pred, target=post_true, reduction="none", log_target=False).sum(dim=-1)
        loss_seq = (kldiv * mask_generate).sum() / denom

        loss_dict = {"rot": loss_rot, "pos": loss_pos, "seq": loss_seq}

        # Auto Energy Matching (positions) — single-pass (no 2nd encoder call)
        if em_active and (E is not None):
            grad_E = torch.autograd.grad(
                E.sum(), p_in,
                create_graph=True,   # required to train E via gradient matching
                retain_graph=True
            )[0]  # (N,L,3)
            grad_E = grad_E * mask_generate[..., None].to(grad_E.dtype)

            # scale factor sqrt(1 - alpha_bar_t)
            alpha_bar = self.trans_pos.var_sched.alpha_bars[t].view(N, 1, 1).clamp(min=EPS, max=1.0)
            c1 = torch.sqrt(1.0 - alpha_bar).clamp(min=EPS)

            # target
            if self.em_use_pred_target:
                target = eps_p_pred.detach() / c1
            else:
                target = eps_p.detach() / c1

            em_raw = F.mse_loss(grad_E, target, reduction="none").sum(dim=-1)  # (N,L)
            em_raw = (em_raw * mask_generate).sum() / denom

            anchor_raw = self._get_anchor_value(loss_dict)

            with torch.no_grad():
                b = self.em_ema_beta
                self.ema_anchor.mul_(b).add_((1.0 - b) * anchor_raw.clamp(min=EPS))
                self.ema_em.mul_(b).add_((1.0 - b) * em_raw.detach().clamp(min=EPS))

                lam = self.em_target_ratio * (self.ema_anchor / self.ema_em)
                lam = torch.clamp(lam, 0.0, self.em_lambda_max)
                lam = lam * self._warmup_factor()
                self.em_step.add_(1)

                self._last_extra_stats = getattr(self, "_last_extra_stats", {})
                self._last_extra_stats.update({
                    "em_raw": float(em_raw.detach().item()),
                    "em_lambda": float(lam.detach().item()),
                })

            loss_dict["em"] = lam.detach() * em_raw

        return loss_dict

    @torch.no_grad()
    def sample(
        self,
        v, p, s,
        res_feat, pair_feat,
        mask_generate, mask_res,
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Reverse sampling over sequence and structure.
        Non-generated residues are held fixed throughout
        """
        N, L = v.shape[:2]
        device = self._dummy.device
        p = self._normalize_position(p)

        # Precompute for masking p-updates
        mask3 = mask_generate[..., None]  # (N,L,1) bool

        # init
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask3, v_rand, v)
            p_init = torch.where(mask3, p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}

        # loop
        if pbar:
            pbar_fn = functools.partial(tqdm, total=self.num_steps, desc="Sampling")
        else:
            pbar_fn = lambda x: x

        for t in pbar_fn(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)

            beta = self.trans_pos.var_sched.betas[t].expand([N])
            t_tensor = torch.full([N], fill_value=t, dtype=torch.long, device=device)

            v_pred, _, eps_p, c_denoised, _ = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )

            v_next = self.trans_rot.denoise(v_t, v_pred, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t - 1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])

        return traj

