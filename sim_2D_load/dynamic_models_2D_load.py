import math
import numpy as np

try:
  import cupy as cp
except Exception:
  cp = None

G = 9.81

# ------------------------ Motor Dynamics ------------------------
class MotorDynamics:
  """
  Inputs:     (Dω_F, Dω_φ)
  Outputs: ω_i, T_i, M_i  (i=1,2)
  """

  def __init__(self,
               kf_in_rpm=6.11e-8,           # [N/(rpm)^2]
               km_in_rpm=1.5e-9,            # [N*m/(rpm)^2]
               motor_gain=20.0,             # [1/s] first-order motor model
               max_omega_rpm=7800.0,        # [rpm]
               use_gpu=False):
    self.xp = (cp if (use_gpu and cp is not None) else np)

    # Unit conversion: rpm -> rad/s (gains for ω^2)
    conv = (30.0 / math.pi) ** 2
    self.kf = kf_in_rpm * conv
    self.km = km_in_rpm * conv

    self.motor_gain = float(motor_gain)
    self.omega_max = (max_omega_rpm * 2.0 * math.pi) / 60.0  # [rad/s]

    # States
    self.omega = self.xp.zeros(2, dtype=float)  # [rad/s]

    # 2x2 mixer for desired motor speeds from (ω_h + Dω_F, Dω_φ)
    # [ω1_des, ω2_des]^T = B @ [ω_h + Dω_F, Dω_φ]^T
    self.B = self.xp.array([[1.0, -1.0],
                            [1.0,  1.0]], dtype=float)

  # --- mixing helpers ---
  def _mix_from_Dw(self, Domega_F: float, Domega_theta: float):
    v = self.xp.array([Domega_F, Domega_theta], dtype=float)
    return self.B @ v  # ω_des (2,)

  def update(self, dt, *, Dw=None):
    """
    Update motor states.
    Either:
      Dw=(Dω_F, Dω_φ)         -- linearized channel inputs, OR
    """
    Domega_F, Domega_theta = Dw # type: ignore
    omega_des = self._mix_from_Dw(Domega_F, Domega_theta)

    # First-order motor dynamics
    self.omega += self.motor_gain * (omega_des - self.omega) * dt
    self.omega = self.xp.clip(self.omega, 0.0, self.omega_max)

    # Individual thrusts / drag torques
    T_i = self.kf * (self.omega ** 2)   # [N]
    M_i = self.km * (self.omega ** 2)   # [N*m]
    return self.omega.copy(), T_i, M_i


# --------------------------- Rigid Body (2D) ---------------------------
class RigidBody2D:
  """
  2D rigid body with one rotational DOF (φ around z), planar translation (x,y) and with a simple pendulum as payload.
  Conventions:
    - Body thrust acts along +y_B (so F_b = [0, ΣT_i]).
    - φ is CCW, R(φ) maps body->world.
    - θ is CCW, angular rotation of the pendulum.
  """

  def __init__(self, drone_mass=0.5, L=0.25, Izz=5e-3,
               x0=0.0, y0=0.0, phi0=0.0, vx0=0.0, vy0=0.0, omega_d0=0.0,
               d=0.1, load_mass=0.1,
               theta0=0.0, omega_l0=0.0,
               use_gpu=False):
    self.xp = (cp if (use_gpu and cp is not None) else np)

    # Drone parameters
    self.m_d = float(drone_mass)
    self.L = float(L)
    self.I = float(Izz)
    # Paiload parameters
    self.m_l = float(load_mass)
    self.d = float(d)

    # States (scalars)
    self.x = float(x0)
    self.y = float(y0)
    self.phi = float(phi0)     # [rad]
    self.theta = float(theta0)     # [rad]
    self.vx = float(vx0)
    self.vy = float(vy0)
    self.omega_d = float(omega_d0)     # [rad/s]
    self.omega_l = float(omega_l0)     # [rad/s]

    # Diagnostics
    self.lin_acc = np.zeros(2, dtype=float)
    self.ang_acc_d = 0.0
    self.ang_acc_l = 0.0

    self.M = self.m_d + self.m_l
    self.a = self.m_l * self.d
    self.b = self.a / self.M

  def state(self):
    x_l = self.x + self.d * math.sin(self.theta)
    y_l = self.y - self.d * math.cos(self.theta)
    vx_l = self.vx + self.d * self.omega_l * math.cos(self.theta)
    vy_l = self.vy + self.d * self.omega_l * math.sin(self.theta)

    return {
      "drone": {
        "position": np.array([self.x, self.y], dtype=float),
        "velocity": np.array([self.vx, self.vy], dtype=float),
        "lin_acc": np.array(self.lin_acc, dtype=float),
        "angle": self.phi,
        "omega": self.omega_d,
      },
      "load":{
        "position": np.array([x_l, y_l], dtype=float),
        "velocity": np.array([vx_l, vy_l], dtype=float),
        "angle": self.theta,
        "omega": self.omega_l,
      }
    }

  def impulse_on_load(self, dtheta=0.0, domega=0.0):
    """
    Aplica un impulso instantáneo al péndulo:
    - dtheta: incremento de ángulo [rad]
    - domega: incremento de velocidad angular [rad/s]
    """
    # Ajusta estos nombres a tu implementación real:
    self.theta += dtheta
    self.omega_l += domega

  def update(self, dt, T_i):
    c_theta, s_theta = math.cos(self.theta), math.sin(self.theta)
    c_phi, s_phi = math.cos(self.phi), math.sin(self.phi)
    sum_F = float(T_i.sum())

    # Drone translational movement
    ax = self.b * self.omega_l**2 * s_theta \
      - sum_F * s_phi / self.M \
      -self.b * self.ang_acc_l * c_theta
    ay = sum_F * c_phi / self.M \
      -self.b * self.ang_acc_l * s_theta \
      - self.b *  self.omega_l**2 * c_theta \
      - G
    self.vx += ax * dt
    self.vy += ay * dt
    self.x += self.vx * dt
    self.y += self.vy * dt
    if self.y < 0.0:
        self.y = 0.0
        self.vy = 0.0
        ay = 0
    self.lin_acc = np.array([ax, ay], dtype=float)

    # Drone rotational movement
    self.ang_acc_d = self.L * (T_i[1] - T_i[0]) / self.I
    self.omega_d += self.ang_acc_d * dt
    self.phi += self.omega_d * dt

    # Pendulum movement
    self.ang_acc_l = -(ax * c_theta + ay * s_theta) / self.d \
      - G * s_theta / self.d
    self.omega_l += self.ang_acc_l * dt
    self.theta += self.omega_l * dt

    return self.state()

