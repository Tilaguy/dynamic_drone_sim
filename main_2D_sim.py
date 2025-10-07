# ========= 2D Drone: Attitude + Motors + Rigid Body (Self-contained) =========
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

try:
    import cupy as cp
except Exception:
    cp = None

G = 9.81

# ------------------------ Motor Dynamics (2D, bi-rotor) ------------------------
class MotorDynamics2D:
    """
    2D bi-rotor motor dynamics + mixer (hover linearized).
    Inputs (Dw mode):     (Dω_F, Dω_θ)
    Inputs (forces mode): (ΔF, Δτ)
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

        # Unit conversion: rpm -> rad/s (gains for ω^2)
        conv = (30.0 / math.pi) ** 2
        self.kf = kf_in_rpm * conv
        self.km = km_in_rpm * conv

        self.L = float(L)
        self.motor_gain = float(motor_gain)
        self.omega_max = (max_omega_rpm * 2.0 * math.pi) / 60.0  # [rad/s]

        # States
        self.omega = self.xp.zeros(2, dtype=float)  # [rad/s]
        self.omega_h = 0.0  # hover baseline [rad/s], set after mass is known

        # 2x2 mixer for desired motor speeds from (ω_h + Dω_F, Dω_θ)
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


# --------------------------- Rigid Body (2D) ---------------------------
class RigidBody2D:
    """
    2D rigid body with one rotational DOF (θ about z) and planar translation (x,y).
    Conventions:
      - Body thrust acts along +y_B (so F_b = [0, ΣT_i]).
      - θ is CCW, R(θ) maps body->world.
    """

    def __init__(self, mass, L=0.25, Izz=5e-3, use_gpu=False,
                 x0=0.0, y0=0.0, theta0=0.0, vx0=0.0, vy0=0.0, omega0=0.0):
        self.xp = (cp if (use_gpu and cp is not None) else np)
        self.m = float(mass)
        self.L = float(L)
        self.I = float(Izz)
        # States (scalars)
        self.x = float(x0)
        self.y = float(y0)
        self.theta = float(theta0)     # [rad]
        self.vx = float(vx0)
        self.vy = float(vy0)
        self.omega = float(omega0)     # [rad/s]

        # Diagnostics
        self.lin_acc = np.zeros(2, dtype=float)
        self.ang_acc = 0.0

    @staticmethod
    def _R(theta: float):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s],
                         [s,  c]], dtype=float)

    def state(self):
        return {
            "position": np.array([self.x, self.y], dtype=float),
            "velocity": np.array([self.vx, self.vy], dtype=float),
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

        # Planar torque from thrust difference (motors at x = ±L)
        tau = self.L * (T_i[1] - T_i[0])

        # Angular dynamics (scalar inertia)
        self.ang_acc = tau / self.I
        self.omega += self.ang_acc * dt
        self.theta += self.omega * dt

        return self.state()


# ----------------------- Attitude Control (2D) -----------------------
class AttitudeControl2D:
    """
    2D attitude PD: outputs (Dω_F, Dω_θ) to feed MotorDynamics2D.
    τ_des = Kp*e_θ - Kd*ω
    Dω_θ  = τ_des / (4*L*kf*ω_h)
    Dω_F  = provided by altitude loop (here PD on y)
    """

    def __init__(self, L, kf, omega_hover,
                 kp_theta=4.0, kd_theta=2.0):
        self.L   = float(L)
        self.kf  = float(kf)           # N/(rad/s)^2
        self.ω_h = float(omega_hover)  # rad/s
        self.kpθ = float(kp_theta)
        self.kdθ = float(kd_theta)
        self._denom_θ = 4.0 * self.L * self.kf * max(self.ω_h, 1e-9)

    def attitude_channels(self, θ_des, θ, ω):
        e_θ = float(θ_des - θ)
        τ_des = self.kpθ * e_θ - self.kdθ * float(ω)
        Dω_θ = τ_des / self._denom_θ
        return Dω_θ


# ----------------------- Simple Altitude PD (2D) -----------------------
class AltitudePD2D:
    """Generates ΔF -> Dω_F = ΔF / (4 kf ω_h)."""
    def __init__(self, kf, omega_hover, kp=8.0, kd=3.0):
        self.kf = float(kf)
        self.ω_h = float(omega_hover)
        self.kp = float(kp)
        self.kd = float(kd)
        self._den = 4.0 * self.kf * max(self.ω_h, 1e-9)

    def channel(self, y_des, y, vy):
        e_y = float(y_des - y)
        ΔF = self.kp * e_y - self.kd * float(vy)  # PD in world y
        Dω_F = ΔF / self._den
        return Dω_F


# ============================= Simulation Setup =============================
# Physical / model params
mass = 0.5
kf_rpm = 6.11e-8
km_rpm = 1.5e-9
L = 0.25
max_omega_rpm = 7800.0
USE_GPU = False

dt = 0.01
sim_time = 10.0
steps = int(sim_time / dt)
sub_steps = 5
sub_dt = dt / sub_steps

# Desired references (sliders will override during run)
desired_y   = 1.0
desired_th  = 0.0  # rad

# Instantiate blocks
drone  = RigidBody2D(mass=mass, L=L, Izz=5e-3, use_gpu=USE_GPU)
motors = MotorDynamics2D(L=L, kf_in_rpm=kf_rpm, km_in_rpm=km_rpm,
                         motor_gain=20.0, max_omega_rpm=max_omega_rpm, use_gpu=USE_GPU)
ω_h = motors.set_hover_from_mass(mass)
att  = AttitudeControl2D(L=L, kf=motors.kf, omega_hover=ω_h, kp_theta=6.0, kd_theta=2.0)
alt  = AltitudePD2D(kf=motors.kf, omega_hover=ω_h, kp=10.0, kd=4.0)

print("Hover ω [rad/s] =", ω_h)

# History buffers
history_len = 300
time_hist = np.arange(history_len)
omega_hist = [np.zeros(history_len), np.zeros(history_len)]  # two motors
wf_hist = np.zeros(history_len)
theta_hist = np.zeros(history_len)
theta_d_hist = np.zeros(history_len)
write_idx = 0

def to_cpu(arr):
    try:
        return arr.get()
    except Exception:
        return arr

# ============================= Plot/Animation =============================
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.35, wspace=0.35, hspace=0.35)

# (0,0) Drone in 2D (x-y)
ax_drone = axs[0,0]
ax_drone.set_xlim(-1.5, 1.5)
ax_drone.set_ylim(0.0, 2.0)
ax_drone.set_xlabel("x [m]")
ax_drone.set_ylabel("y [m]")
ax_drone.set_title("2D Drone (x-y)")
drone_body, = ax_drone.plot([], [], "o-", lw=3, markersize=10)

# (0,1) Motor speeds over time
ax_mot = axs[0,1]
ax_mot.set_title("Motor speeds ω_i(t)")
ax_mot.set_xlim(0, history_len)
ax_mot.set_ylim(0, 1.1 * (max_omega_rpm * 2.0 * math.pi) / 60.0)
ax_mot.set_xlabel("step")
ax_mot.set_ylabel("ω [rad/s]")
lines_motors = [ax_mot.plot([], [])[0] for _ in range(2)]
ax_mot.legend([f"M{i+1}" for i in range(2)], loc="upper right")

# (1,0) Dω_F history (altitude channel)
ax_wf = axs[1,0]
ax_wf.set_title("Altitude channel Dω_F")
ax_wf.set_xlim(0, history_len)
ax_wf.set_ylim(-200.0, 200.0)
ax_wf.set_xlabel("step")
ax_wf.set_ylabel("Dω_F [rad/s]")
line_wf, = ax_wf.plot([], [])

# (1,1) θ vs θ_des
ax_th = axs[1,1]
ax_th.set_title("Attitude: θ and θ_des")
ax_th.set_xlim(0, history_len)
ax_th.set_ylim(-0.6, 0.6)
ax_th.set_xlabel("step")
ax_th.set_ylabel("angle [rad]")
line_theta, = ax_th.plot([], [])
line_theta_d, = ax_th.plot([], [])
ax_th.legend(["θ", "θ_des"], loc="upper right")

# Initialize lines
y_blank = np.full(history_len, np.nan)
for i in range(2):
    lines_motors[i].set_data(time_hist, y_blank.copy())
line_wf.set_data(time_hist, y_blank.copy())
line_theta.set_data(time_hist, y_blank.copy())
line_theta_d.set_data(time_hist, y_blank.copy())

# Sliders
ax_sliders = plt.axes([0.25, 0.05, 0.65, 0.25])
ax_sliders.axis("off")
s_y  = Slider(plt.axes([0.30, 0.22, 0.55, 0.03]), "y_d [m]",   0.0, 1.8, valinit=desired_y)
s_th = Slider(plt.axes([0.30, 0.17, 0.55, 0.03]), "θ_d [rad]", -0.5, 0.5, valinit=desired_th)
s_y.on_changed(lambda val: globals().__setitem__('desired_y', s_y.val))
s_th.on_changed(lambda val: globals().__setitem__('desired_th', s_th.val))

def plots_update(state, W_i, Dω_F):
    global write_idx
    # Drone body segment in world (x-y)
    θ = float(state["theta"])
    x = float(state["position"][0])
    y = float(state["position"][1])
    half_L = L
    x1 = x - half_L * math.cos(θ)
    y1 = y - half_L * math.sin(θ)
    x2 = x + half_L * math.cos(θ)
    y2 = y + half_L * math.sin(θ)
    drone_body.set_data([x1, x2], [y1, y2])

    # Histories
    for i in range(2):
        omega_hist[i][write_idx] = float(to_cpu(W_i[i]))
        lines_motors[i].set_ydata(omega_hist[i])

    wf_hist[write_idx] = float(Dω_F)
    line_wf.set_ydata(wf_hist)

    theta_hist[write_idx]   = float(state["theta"])
    theta_d_hist[write_idx] = float(desired_th)
    line_theta.set_ydata(theta_hist)
    line_theta_d.set_ydata(theta_d_hist)

    write_idx = (write_idx + 1) % history_len

last_time = time.perf_counter()

def update(frame):
    global last_time
    # Current state
    state = drone.state()

    # Channels from controllers
    Dω_F = alt.channel(desired_y, state["position"][1], state["velocity"][1])
    Dω_θ = att.attitude_channels(desired_th, state["theta"], state["omega"])

    # Substeps for numerical stability
    for _ in range(sub_steps):
        W_i, T_i, M_i = motors.update(sub_dt, Dw=(Dω_F, Dω_θ))
        state = drone.update(sub_dt, T_i)

    # Update plots occasionally
    if frame % 1 == 0:
        plots_update(state, W_i, Dω_F)

    # Real-time pacing (optional)
    elapsed = time.perf_counter() - last_time
    last_time = time.perf_counter()
    sleep_time = dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

    return drone_body, *lines_motors, line_wf, line_theta, line_theta_d

def init():
    return (drone_body, *lines_motors, line_wf, line_theta, line_theta_d)

ani = animation.FuncAnimation(fig, update, frames=steps, blit=True, interval=dt*1000, init_func=init)
for ln in [drone_body, *lines_motors, line_wf, line_theta, line_theta_d]:
    ln.set_animated(True)
for i in range(2):
    lines_motors[i].set_xdata(time_hist)
line_wf.set_xdata(time_hist)
line_theta.set_xdata(time_hist)
line_theta_d.set_xdata(time_hist)

plt.show()
