# --------------------------------------------------------
# DPS-RF MAR: Diffusion Posterior Sampling with Rectified Flow
#
# Corrected ODE:
#   dz/dt = v_θ(z_t, t) - ζ(t) · ∇_{z_t} L / ||∇_{z_t} L||
#
# Supports the first-order Euler and second-order Heun ODE solvers.
# Heun applies a full second-order approximation to the corrected ODE by
# evaluating RF velocity and guidance at (z, t) and (z_euler, t_next), then averaging.
#
# Key point: guidance corrects z (the ODE trajectory) directly instead of
# correcting x0 and then interpolating.
#   Wrong (v1 bug): x0_guided = x0 - zeta*grad; z = interp(z, x0_guided)
#     -> An interpolation ratio near 0.02 discards 98% of the correction.
#   Correct (DPS): z = interp(z, x0) - zeta*grad
#     -> Corrections act directly on the trajectory and accumulate over time.
#
# Post-processing strategy:
#   DPS output is artifact-free but blurry because of diffusion-model averaging.
#   One-step sinogram replacement plus FBP restores detail but can add secondary
#   artifacts by amplifying sinogram discontinuities.
#   Progressive DC refinement starts from DPS output and approaches data consistency
#   through multiple small gradient steps without abrupt FBP streaks.
#   The step count controls detail recovery; early stopping acts as regularization.
#
# Reference:
#   Chung et al., "Diffusion Posterior Sampling for General Noisy
#   Inverse Problems", NeurIPS 2023
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np


class DPSRF:
    """
    Diffusion Posterior Sampling with Rectified Flow for MAR.

    RF convention:
      z_t = t · x + (1-t) · ε,  t ∈ [0, 1]
      t = 0: pure noise; t = 1: clean image
    """

    def __init__(
        self,
        num_steps=100,
        guidance_scale=5.0,
        noise_scale=1.0,
        guidance_schedule='linear',
        start_guidance_at=0.05,
        max_zeta=5.0,
        grad_mode='full',
        init_mode='noise',
        init_t=0.0,
        sampling_method='heun',           # ODE solver: second-order Heun or first-order Euler
        # ---- Post-processing: progressive DC refinement ----
        final_refine_mode='gradient_dc',  # 'none', 'sino_replace', 'gradient_dc', 'hybrid_mask'
        final_dc_steps=50,                # More refinement steps are sharper but may add artifacts.
        final_dc_eta=0.01,                # Maximum per-pixel correction per step
        # ---- hybrid_mask options ----
        final_mask_dilate=5,              # 3x3 metal-mask dilation count; about 1 px per iteration
        hybrid_mask_mode='original',      # 'original' or 'projection_augmented'
        hybrid_proj_dilate=3,             # Sinogram metal-trace dilation count
        hybrid_proj_threshold=0.15,       # Threshold for binarizing the backprojected risk map
    ):
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.noise_scale = noise_scale
        self.guidance_schedule = guidance_schedule
        self.start_guidance_at = start_guidance_at
        self.max_zeta = max_zeta
        self.grad_mode = grad_mode
        self.init_mode = init_mode
        self.init_t = init_t
        self.sampling_method = sampling_method
        self.final_refine_mode = final_refine_mode
        self.final_dc_steps = final_dc_steps
        self.final_dc_eta = final_dc_eta
        self.final_mask_dilate = final_mask_dilate
        self.hybrid_mask_mode = hybrid_mask_mode
        self.hybrid_proj_dilate = hybrid_proj_dilate
        self.hybrid_proj_threshold = hybrid_proj_threshold
        self.last_used_hybrid_mask = None
        self.last_projection_risk_map = None

    # ---- Normalization ----

    def _denorm_image(self, x):
        """[-1,1] -> [0,1]"""
        return (x + 1.) / 2.

    def _norm_image(self, x):
        """[0,1] -> [-1,1]"""
        return x * 2. - 1.

    def _denorm_sino(self, s):
        """[-1,1] -> [0,4]"""
        return s * 2. + 2.

    # ---- Guidance-weight schedule ----

    def _guidance_weight(self, t):
        """Return time-dependent guidance strength zeta(t), clipped to its upper bound."""
        if t < self.start_guidance_at:
            return 0.0

        if self.guidance_schedule == 'constant':
            zeta = self.guidance_scale
        elif self.guidance_schedule == 'linear':
            zeta = self.guidance_scale * t
        elif self.guidance_schedule == 'cosine':
            progress = (t - self.start_guidance_at) / (1.0 - self.start_guidance_at)
            zeta = self.guidance_scale * (1 - np.cos(progress * np.pi / 2))
        else:
            zeta = self.guidance_scale * t

        return min(zeta, self.max_zeta)

    # ---- Likelihood gradient (full DPS) ----

    def _compute_guidance_full(self, z, t_batch, labels, model,
                               known_sino, mask_sino, fp, S_known_phys):
        """Compute full DPS gradient dL/dz_t by backpropagating through the model."""
        z_var = z.detach().requires_grad_(True)
        x_pred = model(z_var, t_batch, labels)

        x_phys = self._denorm_image(x_pred)
        S_pred = fp(x_phys)
        diff = mask_sino * (S_pred - S_known_phys)
        loss = (diff ** 2).mean()

        grad = torch.autograd.grad(loss, z_var)[0]

        return grad, x_pred.detach(), loss.item()

    # ---- Likelihood gradient (approximate DPS) ----

    def _compute_guidance_approx(self, z, t_batch, labels, model,
                                 known_sino, mask_sino, fp, S_known_phys):
        """Compute approximate dL/dx0 through fp only to reduce VRAM use."""
        with torch.no_grad():
            x_pred = model(z, t_batch, labels)

        x_var = x_pred.detach().requires_grad_(True)
        x_phys = self._denorm_image(x_var)
        S_pred = fp(x_phys)
        diff = mask_sino * (S_pred - S_known_phys)
        loss = (diff ** 2).mean()

        loss.backward()
        grad = x_var.grad.detach()

        return grad, x_pred, loss.item()

    # ---- ODE stepping ----

    def _rf_step(self, z, t, t_next, x_pred):
        """Take a numerically stable RF x-prediction step without dividing by 1-t."""
        if t_next >= 1.0:
            return x_pred
        ratio = (1 - t_next) / (1 - t)
        return ratio * z + (1 - ratio) * x_pred

    def _evaluate_dps_velocity(self, z, t_val, shape, model,
                                known_sino, mask_sino, fp, S_known_phys,
                                compute_guidance, device):
        """
        Evaluate the full corrected ODE velocity field at (z, t).

        Returns:
            x_pred: Clean image predicted by the model.
            guidance_correction: zeta(t) * grad(L) / ||grad(L)|| in z space,
                or zero when guidance is disabled.
            loss_val: Data-consistency loss used for monitoring.
        """
        t_batch = torch.full((shape[0],), t_val, device=device)
        labels = torch.full((shape[0],), 1000, device=device, dtype=torch.long)

        zeta = self._guidance_weight(t_val)

        if zeta > 0:
            grad, x_pred, loss_val = compute_guidance(
                z, t_batch, labels, model,
                known_sino, mask_sino, fp, S_known_phys
            )
            grad_norm = grad.norm() + 1e-8
            guidance_correction = zeta * (grad / grad_norm)
        else:
            with torch.no_grad():
                x_pred = model(z, t_batch, labels)
            guidance_correction = torch.zeros_like(z)
            loss_val = -1.0

        return x_pred, guidance_correction, loss_val

    # ---- Post-processing 1: one-step sinogram replacement ----

    def _final_sino_replacement(self, x, known_sino, mask_sino, fp, bp):
        """Restore detail with one-step sinogram replacement plus FBP."""
        with torch.no_grad():
            S_pred = fp(self._denorm_image(x))
            S_known_phys = self._denorm_sino(known_sino).clamp(0, 4)
            S_replaced = mask_sino * S_known_phys + (1 - mask_sino) * S_pred
            x_replaced = self._norm_image(bp(S_replaced))
        return x_replaced

    # ---- Post-processing 2: progressive gradient-based DC refinement ----

    def _final_gradient_dc(self, x, known_sino, mask_sino, fp):
        """
        Apply progressive data-consistency refinement.

        Starting from DPS output, use multiple small gradient-descent updates to
        approach sinogram consistency without the abrupt streaks of one-step FBP.

        Physical intuition:
          DPS output is artifact-free but blurry and lacks pixel-level detail.
          The known sinogram contains correct detail from non-metal rays.
          Gradient descent injects this detail into the image incrementally.

        Early stopping acts as implicit regularization:
          Fewer steps favor DPS and produce clean but blurry output.
          More steps favor data consistency and produce sharp output with possible artifacts.
          An intermediate count provides a clean, sharp balance.
        """
        S_known_phys = self._denorm_sino(known_sino).clamp(0, 4)

        for step in range(self.final_dc_steps):
            x_var = x.detach().requires_grad_(True)
            x_phys = self._denorm_image(x_var)
            S_pred = fp(x_phys)

            diff = mask_sino * (S_pred - S_known_phys)
            loss = (diff ** 2).mean()

            loss.backward()
            grad = x_var.grad.detach()

            # Max normalization lets eta directly control the largest pixel update.
            grad_max = grad.abs().max() + 1e-8
            x = x - self.final_dc_eta * (grad / grad_max)
            x = x.clamp(-1, 1)

            if step % max(self.final_dc_steps // 5, 1) == 0 or step == self.final_dc_steps - 1:
                print(f"    DC refine step {step}/{self.final_dc_steps}: "
                      f"dc_loss={loss.item():.6e}")

        return x

    # ---- Post-processing 3: hybrid-mask mode ----

    @staticmethod
    def _dilate_mask(mask, iterations):
        """
        Apply max-pool-based binary dilation in a GPU-friendly Torch implementation.

        Args:
            mask: [1,1,H,W] float32; 1 for metal and 0 elsewhere.
            iterations: Number of 3x3 dilations, each expanding about one pixel.
        Returns:
            dilated: [1,1,H,W] bool
        """
        m = mask.float()
        for _ in range(iterations):
            m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
        return m > 0.5

    def _projection_augmented_mask(self, mask_sino, bp_raw, ref_tensor):
        """
        Build an additional uncertain region by unfiltered backprojection of the
        sinogram-domain metal trace.

        Procedure:
          1. Recover the metal trace from mask_sino: trace = 1 - mask_sino.
          2. Dilate it slightly in sinogram space to cover nearby affected rays.
          3. Map it into image space with unfiltered backprojection.
          4. Normalize by coverage and threshold the result.
        """
        if bp_raw is None:
            raise ValueError(
                "hybrid_mask_mode='projection_augmented' requires the bp_raw "
                "(unfiltered backprojection) argument"
            )

        trace_sino = (1.0 - mask_sino).clamp(0.0, 1.0).float()
        if self.hybrid_proj_dilate > 0:
            trace_sino = self._dilate_mask(trace_sino, self.hybrid_proj_dilate).float()

        with torch.no_grad():
            proj_risk = bp_raw(trace_sino).abs()
            coverage = bp_raw(torch.ones_like(trace_sino)).abs()
            proj_risk = proj_risk / (coverage + 1e-8)
            proj_risk = proj_risk / (proj_risk.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        proj_mask = proj_risk >= self.hybrid_proj_threshold
        self.last_projection_risk_map = proj_risk.detach().to(ref_tensor.device)
        return proj_mask

    def _build_hybrid_mask(self, metal_mask, mask_sino, bp_raw, ref_tensor):
        """Build the final image-domain binary mask used by hybrid_mask."""
        metal_mask = metal_mask.to(ref_tensor.device)
        base_mask = self._dilate_mask(metal_mask, self.final_mask_dilate)

        if self.hybrid_mask_mode == 'original':
            hybrid_mask = base_mask
        elif self.hybrid_mask_mode == 'projection_augmented':
            proj_mask = self._projection_augmented_mask(mask_sino, bp_raw, ref_tensor)
            hybrid_mask = torch.logical_or(base_mask, proj_mask)
        else:
            raise ValueError(
                f"Unknown hybrid_mask_mode: {self.hybrid_mask_mode}. "
                "Options: 'original', 'projection_augmented'"
            )

        self.last_used_hybrid_mask = hybrid_mask.float().detach().to(ref_tensor.device)
        return hybrid_mask

    def _final_hybrid_mask(self, x, known_sino, mask_sino, fp, bp, bp_raw, metal_mask):
        """
        Apply hybrid-mask post-processing.

        Strategy:
          - Dilate the metal mask final_mask_dilate times to cover nearby artifacts.
          - Inside the mask, use progressive gradient_dc for smooth repair without FBP streaks.
          - Outside the mask, use one-step sino_replace for sharp normal tissue.
          - Blend as result = dilated_mask * x_dc + (1 - dilated_mask) * x_sino.

        Physical intuition:
          sino_replace works well far from metal because the sinogram is already clean,
          but near metal, FBP amplifies discontinuities into streaks. gradient_dc adds
          data consistency gradually near metal. Spatial blending combines both strengths.
        """
        dilated = self._build_hybrid_mask(
            metal_mask, mask_sino, bp_raw=bp_raw, ref_tensor=x
        ).float()

        print(f"  Hybrid mask ({self.hybrid_mask_mode}): "
              f"dilate={self.final_mask_dilate}  "
              f"mask_ratio={dilated.mean().item():.3f}")
        if self.hybrid_mask_mode == 'projection_augmented':
            print(f"    projection_augmented: sino_dilate={self.hybrid_proj_dilate}  "
                  f"threshold={self.hybrid_proj_threshold:.3f}")

        # Outside the mask: exact one-step sinogram replacement.
        x_sino = self._final_sino_replacement(x, known_sino, mask_sino, fp, bp)

        # Inside the mask: progressive gradient-based DC refinement.
        x_dc = self._final_gradient_dc(x, known_sino, mask_sino, fp)

        # Spatial blending
        result = dilated * x_dc + (1.0 - dilated) * x_sino
        return result.clamp(-1, 1)

    # ---- Main sampling loop ----

    def sample(self, model, shape, known_sino, mask_sino, fp, bp, device,
               bp_raw=None,
               XLI=None, metal_mask=None):
        """
        Run DPS-RF MAR sampling.

        Args:
            model: Pretrained JiT RF model.
            shape: (N, C, H, W)
            known_sino: [-1,1] (Sma)
            mask_sino: Approximately 1 outside metal and 0 inside metal.
            fp: Differentiable forward projection.
            bp: Filtered backprojection.
            bp_raw: Unfiltered backprojection/adjoint for projection-augmented masks.
            device: torch device
            XLI: Linear-interpolation reconstruction in [-1,1], used by xli_blend.
            metal_mask: Image-domain CT metal mask [1,1,H,W], used only by hybrid_mask.
        """
        known_sino   = known_sino.to(device)
        mask_sino    = mask_sino.to(device)
        S_known_phys = self._denorm_sino(known_sino).clamp(0, 4)
        self.last_used_hybrid_mask = None
        self.last_projection_risk_map = None

        # ---- Initialization ----
        if self.init_mode == 'xli_blend' and XLI is not None:
            eps = self.noise_scale * torch.randn(*shape, device=device)
            t0 = self.init_t
            z = t0 * XLI.to(device) + (1 - t0) * eps
            start_step = int(t0 * self.num_steps)
            print(f"  DPS-RF: XLI blend init at t={t0:.2f}, "
                  f"starting from step {start_step}/{self.num_steps}")
        else:
            z = self.noise_scale * torch.randn(*shape, device=device)
            start_step = 0

        timesteps = torch.linspace(0, 1.0, self.num_steps + 1, device=device)

        compute_guidance = (self._compute_guidance_full if self.grad_mode == 'full'
                            else self._compute_guidance_approx)

        use_heun = (self.sampling_method == 'heun')
        print(f"  ODE solver: {self.sampling_method}")

        # ---- ODE + DPS ----
        for i in range(start_step, self.num_steps):
            t_val = timesteps[i].item()
            t_next_val = timesteps[i + 1].item()

            if use_heun:
                z, loss_val = self._heun_dps_step(
                    z, t_val, t_next_val, shape, model,
                    known_sino, mask_sino, fp, S_known_phys,
                    compute_guidance, device
                )
            else:
                z, loss_val = self._euler_dps_step(
                    z, t_val, t_next_val, shape, model,
                    known_sino, mask_sino, fp, S_known_phys,
                    compute_guidance, device
                )

            if i % max(self.num_steps // 10, 1) == 0 or i == self.num_steps - 1:
                zeta = self._guidance_weight(t_val)
                print(f"  DPS step {i}/{self.num_steps}: t={t_val:.3f}, "
                      f"zeta={zeta:.4f}, dc_loss={loss_val:.4e}")

        # ---- Post-processing ----
        result = z.clamp(-1, 1)

        if self.final_refine_mode == 'sino_replace':
            result = self._final_sino_replacement(
                result, known_sino, mask_sino, fp, bp
            )
            print(f"  Final: sinogram replacement")

        elif self.final_refine_mode == 'gradient_dc':
            print(f"  Final: gradient DC refinement "
                  f"({self.final_dc_steps} steps, eta={self.final_dc_eta})")
            result = self._final_gradient_dc(
                result, known_sino, mask_sino, fp
            )

        elif self.final_refine_mode == 'hybrid_mask':
            if metal_mask is None:
                raise ValueError(
                    "final_refine_mode='hybrid_mask' requires the metal_mask argument")
            print(f"  Final: hybrid mask refinement "
                  f"(dilate={self.final_mask_dilate}, "
                  f"dc_steps={self.final_dc_steps}, eta={self.final_dc_eta})")
            result = self._final_hybrid_mask(
                result, known_sino, mask_sino, fp, bp, bp_raw, metal_mask
            )

        else:
            print(f"  Final: none (raw DPS output)")

        return result.clamp(-1, 1)

    def _euler_dps_step(self, z, t_val, t_next_val, shape, model,
                        known_sino, mask_sino, fp, S_known_phys,
                        compute_guidance, device):
        """
        Take a first-order Euler DPS step.

        z_next = rf_step(z, t, t_next, x_pred) - ζ(t) · ∇L/||∇L||
        """
        x_pred, guidance_corr, loss_val = self._evaluate_dps_velocity(
            z, t_val, shape, model,
            known_sino, mask_sino, fp, S_known_phys,
            compute_guidance, device
        )

        with torch.no_grad():
            z_next = self._rf_step(z, t_val, t_next_val, x_pred)
            z_next = z_next - guidance_corr

        return z_next, loss_val

    def _heun_dps_step(self, z, t_val, t_next_val, shape, model,
                       known_sino, mask_sino, fp, S_known_phys,
                       compute_guidance, device):
        """
        Take a second-order Heun DPS step.

        Apply full Heun integration to the corrected ODE
        dz/dt = v_RF(z,t) - zeta(t)*grad(L)/||grad(L)||:
          1. Evaluate x_pred1 and guidance1 at (z, t).
          2. Predict z_euler = rf_step(z, t, t_next, x_pred1) - guidance1.
          3. Evaluate x_pred2 and guidance2 at (z_euler, t_next).
          4. Average both predictions and guidance corrections for the final step:
             z_next = rf_step(z, t, t_next, x_avg) - 0.5*(guidance1 + guidance2)
        """
        # ---- First evaluation at (z, t) ----
        x_pred1, guidance_corr1, loss_val = self._evaluate_dps_velocity(
            z, t_val, shape, model,
            known_sino, mask_sino, fp, S_known_phys,
            compute_guidance, device
        )

        with torch.no_grad():
            # Euler prediction
            z_euler = self._rf_step(z, t_val, t_next_val, x_pred1)
            z_euler = z_euler - guidance_corr1

        # Return the final step directly.
        if t_next_val >= 1.0:
            return z_euler, loss_val

        # ---- Second evaluation at (z_euler, t_next) ----
        x_pred2, guidance_corr2, _ = self._evaluate_dps_velocity(
            z_euler, t_next_val, shape, model,
            known_sino, mask_sino, fp, S_known_phys,
            compute_guidance, device
        )

        with torch.no_grad():
            # Average x predictions for the RF step and guidance for the correction.
            x_avg = 0.5 * (x_pred1 + x_pred2)
            z_next = self._rf_step(z, t_val, t_next_val, x_avg)
            z_next = z_next - 0.5 * (guidance_corr1 + guidance_corr2)

        return z_next, loss_val
