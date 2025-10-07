import math
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

G = 9.81

class MotorDynamics2D:
    """
    2D bi-rotor motor dynamics + mixer (hover linearized).
    Inputs (mode='Dw'):     (Dω_F, Dω_θ)
    Inputs (mode='forces'): (ΔF, Δτ)
    Outputs: ω_i, T_i, M_i  (i=1,2)
    """

    def __init__(self,
                 L=0.25,                      # [m] arm length (±L on x_B)
                 kf_in_rpm=6.11e-8,           # [N/(rpm)^2]
                 km_in_rpm=1.5e-9,            # [N*m/(rpm)^2]
                 motor_gain=20.0,             # [1/s] first-order motor model
                 max_omega_rpm=7800.0,        # [rpm]
                 use_gpu=False):
        self.xp = (cp if (use_gpu and cp is not None) else np)

        # Unit conversion: rpm -> rad/s (squared gains)
        conv = (30.0 / math.pi) ** 2
        self.kf = kf_in_rpm * conv
        self.km = km_in_rpm * conv

        self.L = L
        self.motor_gain = motor_gain
        self.omega_max = (max_omega_rpm * 2.0 * math.pi) / 60.0  # [rad/s]

        # States
        self.omega = self.xp.zeros(2, dtype=float)  # [rad/s]
        self.omega_h = 0.0  # hover baseline [rad/s], set after mass is known

        # 2x2 mixer for desired **motor speeds** from (ω_h + Dω_F, Dω_θ)
        # [ω1_des, ω2_des]^T = B @ [ω_h + Dω_F, Dω_θ]^T
        self.B = self.xp.array([[1.0, -1.0],
                                [1.0,  1.0]], dtype=float)

    def set_hover_from_mass(self, mass: float):
        """Compute hover speed from mass: mg = 2*kf*ω_h^2  ->  ω_h = sqrt(mg/(2*kf))."""
        self.omega_h = math.sqrt(max(mass * G / (2.0 * self.kf), 0.0))
        return self.omega_h

    # --- mixing helpers ---
    def _mix_from_Dw(self, Domega_F: float, Domega_theta: float):
        v = self.xp.array([self.omega_h + Domega_F, Domega_theta], dtype=float)
        return self.B @ v  # ω_des (2,)

    def _mix_from_forces(self, dF: float, dtau: float):
        # From linearization in 2D:
        # ΔF   = 4*kf*ω_h * Dω_F     -> Dω_F   = ΔF / (4*kf*ω_h)
        # Δτ   = 4*L*kf*ω_h * Dω_θ   -> Dω_θ   = Δτ / (4*L*kf*ω_h)
        denomF = 4.0 * self.kf * max(self.omega_h, 1e-9)
        denomT = 4.0 * self.L * self.kf * max(self.omega_h, 1e-9)
        Domega_F    = dF   / denomF
        Domega_theta= dtau / denomT
        return self._mix_from_Dw(Domega_F, Domega_theta)

    def update(self, dt, *, Dw=None, forces=None):
        """
        Update motor states.
        Either:
          Dw=(Dω_F, Dω_θ)         -- linearized channel inputs, OR
          forces=(ΔF, Δτ)         -- physical deviations
        """
        if Dw is not None:
            Domega_F, Domega_theta = Dw
            omega_des = self._mix_from_Dw(Domega_F, Domega_theta)
        elif forces is not None:
            dF, dtau = forces
            omega_des = self._mix_from_forces(dF, dtau)
        else:
            raise ValueError("Provide either Dw=(Dω_F, Dω_θ) or forces=(ΔF, Δτ).")

        # First-order motor dynamics
        self.omega += self.motor_gain * (omega_des - self.omega) * dt
        self.omega = self.xp.clip(self.omega, 0.0, self.omega_max)

        # Individual thrusts / drag torques
        T_i = self.kf * (self.omega ** 2)   # [N]
        M_i = self.km * (self.omega ** 2)   # [N*m]
        return self.omega.copy(), T_i, M_i


class RigidBody2D:
    """
    2D rigid body with one rotational DOF (θ about z) and planar translation (x,y).
    Conventions:
      - Body thrust acts along +y_B (so F_b = [0, ΣT_i]).
      - θ is CCW, R(θ) maps body->world.
    """

    def __init__(self, mass, Izz=5e-3, use_gpu=False, L=0.25,
                 x0=0.0, y0=0.0, theta0=0.0, vx0=0.0, vy0=0.0, omega0=0.0):
        self.xp = (cp if (use_gpu and cp is not None) else np)
        self.m = float(mass)
        self.I = float(Izz)
        self.L = float(L)
        # States (scalars/vectors)
        self.x = float(x0)
        self.y = float(y0)
        self.theta = float(theta0)     # [rad]
        self.vx = float(vx0)
        self.vy = float(vy0)
        self.omega = float(omega0)     # [rad/s]

        # Diagnostics
        self.lin_acc = self.xp.zeros(2, dtype=float)
        self.ang_acc = 0.0

    @staticmethod
    def _R(theta: float):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s],
                         [s,  c]], dtype=float)

    def state(self):
        return {
            "pos": np.array([self.x, self.y], dtype=float),
            "vel": np.array([self.vx, self.vy], dtype=float),
            "theta": self.theta,
            "omega": self.omega,
            "lin_acc": np.array(self.lin_acc, dtype=float),
            "ang_acc": self.ang_acc
        }

    def update(self, dt, T_i):
        T_i = np.asarray(T_i, dtype=float).reshape(2,)
        # Body force (sum of rotor thrusts along +y_B)
        F_b = np.array([0.0, float(T_i.sum())], dtype=float)
        # Rotate to world and add gravity
        F_w = self._R(self.theta) @ F_b + np.array([0.0, -self.m * G], dtype=float)
        # Linear dynamics
        ax, ay = F_w / self.m
        self.lin_acc[:] = [ax, ay]
        self.vx += ax * dt
        self.vy += ay * dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt

        # Planar torque (motors at x = ±L; pass T_i in same order as MotorDynamics2D)
        # Here we don't need L again if already used to compute τ outside; but for clarity:
        # τ = L*(T2 - T1). If you prefer, compute τ externally and pass it instead.
        # For now, just illustrative; you can remove if you prefer external τ.
        # (Keep consistent with your MotorDynamics2D L if you reuse it here.)
        # We'll assume L known to the caller or compute τ outside and set `tau` here.
        # Example with L=0.25:
        tau = self.L * (T_i[1] - T_i[0])

        # Angular dynamics (no gyroscopic term in 2D with scalar inertia)
        self.ang_acc = tau / self.I
        self.omega += self.ang_acc * dt
        self.theta += self.omega * dt

        return self.state()
