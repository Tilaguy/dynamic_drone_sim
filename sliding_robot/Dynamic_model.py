import math
import numpy as np

G = 9.81
eps = 1e-6

# ------------------------ Motor Dynamics (2D, bi-rotor) ------------------------
class MotorDynamics2D:
    """
    2D bi-rotor motor dynamics + mixer (hover linearized).
    Inputs (Dw mode):     (Dω_F, Dω_θ)
    Inputs (forces mode): (ΔF, Δτ)
    Outputs: ω_i, T_i, M_i  (i=1,2)
    """

    def __init__(self,
                 L=0.25,                    # [m] arm length (±L on x_B)
                 kf_in_rpm=6.11e-8,        # [N/(rpm)^2]
                 km_in_rpm=1.5e-9,               # [N*m/(rpm)^2]
                 motor_gain=20.0,   # [1/s] first-order motor model
                 max_omega_rpm=7800.0,    # [rpm]
                 ):

        # Unit conversion: rpm -> rad/s (gains for ω^2)
        conv = (30.0 / math.pi) ** 2
        self.kf = kf_in_rpm * conv
        self.km = km_in_rpm * conv

        self.L = L
        self.motor_gain = motor_gain
        self.omega_max = (max_omega_rpm * 2.0 * math.pi) / 60.0 # [rad/s]

        # States
        self.omega = np.zeros(2, dtype=float)   # [rad/s]
        self.omega_h = 0.0  # hover baseline [rad/s], set after mass is known

        # 2x2 mixer for desired motor speeds from (ω_h + Dω_F, Dω_θ)
        # [ω1_des, ω2_des]^T = B @ [ω_h + Dω_F, Dω_θ]^T
        self.B = np.array([[1.0, -1.0],
                           [1.0,    1.0]],
                          dtype=float)

    def set_hover_from_mass(self, mass: float):
        """Compute hover speed from mass: mg = 2*kf*ω_h^2   ->   ω_h = sqrt(mg/(2*kf))."""
        self.omega_h = math.sqrt(max(mass * G / (2.0 * self.kf), 0.0))
        return self.omega_h

    # --- mixing helpers ---
    def _mix_from_Dw(self, Domega_F: float, Domega_theta: float):
        v = np.array([self.omega_h + Domega_F, Domega_theta], dtype=float)
        return self.B @ v   # ω_des (2,)

    def _mix_from_forces(self, dF: float, dtau: float):
        # From linearization in 2D:
        # ΔF     = 4*kf*ω_h * Dω_F     -> Dω_F     = ΔF / (4*kf*ω_h)
        # Δτ     = 4*L*kf*ω_h * Dω_θ   -> Dω_θ     = Δτ / (4*L*kf*ω_h)
        denomF = 4.0 * self.kf * max(self.omega_h, 1e-9)
        denomT = 4.0 * self.L * self.kf * max(self.omega_h, 1e-9)
        Domega_F    = dF     / denomF
        Domega_theta= dtau / denomT
        return self._mix_from_Dw(Domega_F, Domega_theta)

    def update(self, dt, *, Dw=None, forces=None):
        """
        Update motor states.
        Either:
            Dw=(Dω_F, Dω_θ)      -- linearized channel inputs, OR
            forces=(ΔF, Δτ)      -- physical deviations
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
        self.omega = np.clip(self.omega, 0.0, self.omega_max)

        # Individual thrusts / drag torques
        T_i = self.kf * (self.omega ** 2)    # [N]
        M_i = self.km * (self.omega ** 2)    # [N*m]
        return self.omega.copy(), T_i, M_i

# ------------------------ Drone Dynamics (2D, bi-rotor) ------------------------
class DroneDynamic2D:
    def __init__(self, mass:float, inertia:float, ini_pos:np.ndarray, Ts:float) -> None:
        self.m = mass
        self.I = inertia
        self.Ts = Ts

        self.DOF = 2 + 1 # x,y and φ
        self.q = np.zeros((self.DOF, 1), dtype=float)
        self.q[:2, 0] = ini_pos.copy().reshape((2,))
        self.dq = np.zeros((self.DOF, 1))
        self.ddq = np.zeros((self.DOF, 1))

        self.A = np.zeros((self.DOF, self.DOF))
        self.Q = np.zeros((self.DOF, 1))
        self.G = np.zeros((self.DOF, 1))

        self.F_t = np.zeros((2, 1))
        self.τ = 0.0

    def update_q(self, q_ref:np.ndarray, dq_ref:np.ndarray, ddq_ref:np.ndarray):
        self.q[:, 0] = q_ref.copy().reshape((3,))
        self.dq[:, 0] = dq_ref.copy().reshape((3,))
        self.ddq[:, 0] = ddq_ref.copy().reshape((3,))

    @staticmethod
    def _R(theta: float):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s],
                         [s,  c]], dtype=float)

    def step(self, dt, F_b:np.ndarray, tau:float):
        # Rotate to world and add gravity
        F_w = (self._R(self.q[2]) @ F_b) + np.array([[0.0], [-self.m * G]], dtype=float)

        # Linear dynamics
        self.ddq[:2] = F_w / self.m

        # Angular dynamics (scalar inertia)
        self.ddq[2, 0] = tau / self.I

        self.dq += self.ddq * dt
        self.q  += self.dq * dt

        return self.q



# ------------------------ Sliding Pulley Dynamics  ------------------------
class PulleyDynamic2D:
    def __init__(self, mass:float, pulley_pos:np.ndarray, cable_len:float, Ts:float, damping:float = 0.0) -> None:
        self.m = mass
        self.L = cable_len
        self.Ts = Ts
        self.b = damping

        self.l = []

        self.v = np.zeros((2,1))
        self.x = pulley_pos.copy().reshape((2,1))

        self.W = np.array([[0], [- mass * G]])
        self.T = [0.0, 0.0]


    def step(self, x1:np.ndarray, x2:np.ndarray, F_ext:float = 0.0):
        # ---- cable geometry ----
        l1 = max(np.linalg.norm(self.x - x1), eps)
        l2 = max(np.linalg.norm(self.x - x2), eps)

        u1 = (x1 - self.x) / l1
        u2 = (x2 - self.x) / l2

        self.l = [l1, l2]

        # gradient of constraint
        s = u1 + u2

        # ---- compute cable tension ----
        den = max((s.T @ s).item(), eps)
        lam = (s.T @ self.W).item() / den

        Fc = -lam * s

        # cable tension magnitude
        T = max(0.0, -lam)
        self.T = [T, T]

        # ---- external disturbance ----
        disturbance = np.array([[F_ext], [0]])

        grad = u1 + u2
        grad_norm = (grad.T @ grad).item()

        if grad_norm > eps:
            # tangent projection matrix
            P = np.eye(2) - (grad @ grad.T) / grad_norm
        else:
            P = np.eye(2)

        # damping only along allowed motion
        F_damp = -self.b * (P @ self.v)

        # total force
        F = self.W + Fc + disturbance + F_damp

        # ---- dynamics integration ----
        a = F / self.m

        self.v += a * self.Ts
        self.x += self.v * self.Ts

        # ==================================================
        # POSITION CONSTRAINT PROJECTION
        # enforce l1 + l2 = L
        # ==================================================

        l1 = np.linalg.norm(self.x - x1)
        l2 = np.linalg.norm(self.x - x2)

        g = l1 + l2 - self.L

        if abs(g) > 1e-8:
            u1 = (self.x - x1) / max(l1, eps)
            u2 = (self.x - x2) / max(l2, eps)

        grad = u1 + u2
        grad_norm = (grad.T @ grad).item()

        if grad_norm > eps:
            alpha = g / grad_norm
            self.x -= alpha * grad

        # ==================================================
        # VELOCITY PROJECTION
        # enforce (u1+u2)^T v = 0
        # ==================================================

        l1 = max(np.linalg.norm(self.x - x1), eps)
        l2 = max(np.linalg.norm(self.x - x2), eps)
        print(f"\rl1: {l1:.3f}, l2: {l2:.3f}, L: {l1+l2:.2f}, E: {0.5 * self.m * (self.v.T @ self.v).item() + self.m * G * self.x[1,0]:.2f}", end="", flush=True)

        u1 = (self.x - x1) / l1
        u2 = (self.x - x2) / l2

        grad = u1 + u2
        grad_norm = (grad.T @ grad).item()

        if grad_norm > eps:
            # remove velocity component along constraint
            v_corr = (grad.T @ self.v).item() / grad_norm
            self.v -= v_corr * grad

# ------------------------ Sliding Pulley Dynamics  ------------------------
class PulleyDynamic_Lagrange2D:
    def __init__(self, mass:float, pulley_pos:np.ndarray, cable_len:float, Ts:float, damping:float = 0.0) -> None:
        self.m = mass
        self.L = cable_len
        self.Ts = Ts
        self.b = damping

        self.l = []
        self.T = []

        self.M = np.zeros((8, 8))
        idx, idy = np.diag_indices(self.M.shape[0])
        self.M[idx[:2], idy[:2]] = self.m
        self.J = np.zeros((1, 8))
        self.Q = np.zeros((8, 1))
        self.B = np.zeros((8, 1))

        self.G = np.zeros((8, 1))
        self.G[1, 0] = -self.m * G  # Payload gravity
        self.G[3, 0] = -G            # Base 1 gravity
        self.G[6, 0] = -G            # Base 2 gravity

        self.q = np.zeros((8, 1))
        try:
            self.q[:2, 0] = pulley_pos.copy().reshape((2,1))
        except ValueError:
            self.q[:2, 0] = np.array([0, 1])
        self.dq = np.zeros_like(self.q)
        self.ddq = np.zeros_like(self.q)

    def update_q(self, base1:DroneDynamic2D, base2:DroneDynamic2D):
        idx, idy = np.diag_indices(self.M.shape[0])
        self.M[idx[2:4], idy[2:4]] = base1.m
        self.M[idx[4], idy[4]] = base1.I
        self.M[idx[5:7], idy[5:7]] = base2.m
        self.M[idx[7], idy[7]] = base2.I

        self.G[3, 0] = -base1.m * G
        self.G[6, 0] = -base2.m * G

        self.q[2:5, 0] =  base1.q.reshape((3,))
        self.q[5:, 0] =  base2.q.reshape((3,))
        self.dq[2:5, 0] =  base1.dq.reshape((3,))
        self.dq[5:, 0] =  base2.dq.reshape((3,))
        self.ddq[2:5, 0] =  base1.ddq.reshape((3,))
        self.ddq[5:, 0] =  base2.ddq.reshape((3,))

    def step(self, base1:DroneDynamic2D, base2:DroneDynamic2D, F_ext:float = 0.0):
        self.update_q(base1, base2)

        self.Q.fill(0)
        self.Q[0, 0] = F_ext
        self.Q[2:4, 0] = base1.F_t.reshape((2,))
        self.Q[4, 0] = base1.τ
        self.Q[5:7, 0] = base2.F_t.reshape((2,))
        self.Q[7, 0] = base2.τ

        x = self.q[0:2, 0]
        x1 = self.q[2:4, 0]
        x2 = self.q[5:7, 0]

        l1 = max(np.linalg.norm(x - x1), eps)
        l2 = max(np.linalg.norm(x - x2), eps)
        self.l = [l1, l2]

        u1 = (x1 - x) / l1
        u2 = (x2 - x) / l2

        self.J.fill(0)
        self.J[0, 0:2] = -u1 - u2  # Derivative with respect to payload x, y
        self.J[0, 2:4] = u1        # Derivative with respect to Base 1 x1, y1
        self.J[0, 5:7] = u2        # Derivative with respect to Base 2 x2, y2
        # Indices 4 and 7 remain 0 because angles theta1, theta2 don't affect cable length

        dx = self.dq[0:2, 0]
        dx1 = self.dq[2:4, 0]
        dx2 = self.dq[5:7, 0]

        v1 = dx1 - dx               # Relative velocity of base 1 to payload
        v2 = dx2 - dx               # Relative velocity of base 2 to payload

        dl1 = np.dot(u1, v1)
        dl2 = np.dot(u2, v2)

        self.B.fill(0)
        self.B[:2, 0] = self.b * dl1 * u1
        self.B[2:4, 0] = -self.b * dl1 * u1

        du1 = (v1 - u1 * dl1) / l1
        du2 = (v2 - u2 * dl2) / l2

        jdqd = np.dot(du1, v1) + np.dot(du2, v2)

        F_total = self.Q + self.B + self.G

        A_matrix = np.block([
            [self.M, self.J.T],
            [self.J, np.zeros((1, 1))]
        ])

        RHS = np.block([
            [F_total],
            [-jdqd]
        ])

        solution = np.linalg.solve(A_matrix, RHS)

        self.ddq = solution[:8]     # First 8 elements are accelerations
        self.tension = solution[8, 0] # 9th element is the Lagrange multiplier (tension)

        self.T = [self.tension + self.b * dl1, self.tension]
        print(f"\rl1: {l1:.3f}, l2: {l2:.3f}, L: {l1+l2:.2f}, E: {0.5 * self.m * (self.dq.T @ self.dq).item() + self.m * G * self.q[1,0]:.2f}", end="", flush=True)

        self.dq += self.ddq * self.Ts
        self.q += self.dq * self.Ts


