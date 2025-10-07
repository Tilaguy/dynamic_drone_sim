import numpy as np
try:
    import cupy as cp
except Exception:
    cp = None

class AttitudeControl2D:
    """
    2D attitude PD: outputs (Dω_F, Dω_θ) to feed MotorDynamics2D.
    τ_des = Kp*e_θ - Kd*ω
    Dω_θ  = τ_des / (4*L*kf*ω_h)
    Dω_F  = 0  (hover handled by ω_h; put altitude loop here if needed)
    """

    def __init__(self, L, kf, omega_hover,
                 kp_theta=4.0, kd_theta=2.0, use_gpu=False):
        self.xp = (cp if (use_gpu and cp is not None) else np)
        self.L   = float(L)
        self.kf  = float(kf)           # N/(rad/s)^2
        self.ω_h = float(omega_hover)  # rad/s
        self.kpθ = float(kp_theta)
        self.kdθ = float(kd_theta)

        # Precompute denominator from linearization:
        self._denom_θ = 4.0 * self.L * self.kf * max(self.ω_h, 1e-9)

    def update(self, θ_des, θ, ω):
        """
        Inputs:
          θ_des: desired angle [rad] (scalar)
          θ:     current angle [rad]
          ω:     current angular rate [rad/s]
        Returns:
          Dω_F, Dω_θ
        """
        e_θ = float(θ_des - θ)          # small-angle assumption
        τ_des = self.kpθ * e_θ - self.kdθ * float(ω)
        Dω_θ = τ_des / self._denom_θ
        Dω_F = 0.0                      # keep hover by ω_h; add altitude loop if desired
        return self.xp.array([Dω_F, Dω_θ], dtype=float)
