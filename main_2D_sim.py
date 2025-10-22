# ========= 2D Drone: Attitude + Motors + Rigid Body (Self-contained) =========
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

from sim_2D.dynamic_models import RigidBody2D, MotorDynamics2D
from sim_2D.control import PositionCascade2D, AttitudeControl2D

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
sub_steps = 2
sub_dt = dt / sub_steps

# Desired references (sliders will override during run)
desired_x   = 0.0
desired_y   = 1.0
desired_th  = 0.0  # rad
θ_des_prev = 0.0

# Instantiate blocks
drone  = RigidBody2D(mass=mass, L=L, Izz=5e-3, use_gpu=USE_GPU)
motors = MotorDynamics2D(L=L, kf_in_rpm=kf_rpm, km_in_rpm=km_rpm,
                         motor_gain=15.0, max_omega_rpm=max_omega_rpm, use_gpu=USE_GPU)
ω_h = motors.set_hover_from_mass(mass)
# ω_h = 0.0
att = AttitudeControl2D(L=L, kf=motors.kf, omega_hover=ω_h, kp_theta=8.0, kd_theta=3.0)
pos = PositionCascade2D(kp_x=3.0, ki_x=0.1, kd_x=2.8,
                        kp_y=11.5, ki_y=0.0, kd_y=5.4,
                        ax_limit=2.0, ay_limit=3.0,
                        L=L, kf=motors.kf, omega_hover=ω_h)

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
ax_sliders = plt.axes([0.25, 0.05, 0.65, 0.25]) # pyright: ignore[reportArgumentType]
ax_sliders.axis("off")
s_x  = Slider(plt.axes([0.30, 0.22, 0.55, 0.03]), "x_d [m]",   -1.5, 1.5, valinit=desired_x) # pyright: ignore[reportArgumentType]
s_y  = Slider(plt.axes([0.30, 0.17, 0.55, 0.03]), "y_d [m]",   0.0, 1.8, valinit=desired_y) # pyright: ignore[reportArgumentType]
s_th = Slider(plt.axes([0.30, 0.12, 0.55, 0.03]), "θ_d [rad]", -0.5, 0.5, valinit=desired_th) # pyright: ignore[reportArgumentType]
s_x.on_changed(lambda val: globals().__setitem__('desired_x', s_x.val))
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
    global last_time, θ_des_prev

    # Estado actual
    state = drone.state()
    x,  y  = state["position"]
    vx, vy = state["velocity"]
    θ,  ω  = state["theta"], state["omega"]

    ax_ref, ay_ref, θ_des_raw, Dω_F = pos.step(xd=desired_x, x=x, vx=vx,
                                               yd=desired_y, y=y, vy=vy,
                                               m=mass, dt=dt)

    max_dtheta = 2.0  # [rad/s] velocidad máx. de referencia
    dθ_cmd = np.clip(θ_des_raw - θ_des_prev, -max_dtheta*dt, max_dtheta*dt)
    theta_des = np.clip(θ_des_prev + dθ_cmd, -0.5, 0.5)
    θ_des_prev = float(theta_des)

    # Lazo de actitud -> canal diferencial
    Dω_θ = att.attitude_channels(theta_des, θ, ω)

    # Saturaciones de canales (protege ω_des)
    Dω_F = np.clip(Dω_F, -0.5*ω_h, 0.5*ω_h)   # p.ej. ±50% de ω_h
    Dω_θ = np.clip(Dω_θ, -0.5*ω_h, 0.5*ω_h)

    # Subpasos para estabilidad numérica
    for _ in range(sub_steps):
        W_i, T_i, M_i = motors.update(sub_dt, Dw=(Dω_F, Dω_θ))
        state = drone.update(sub_dt, T_i)

    # Plots
    if frame % 1 == 0:
        plots_update(state, W_i, Dω_F)
        # (opcional) refleja la ref. usada en el gráfico de θ_d:
        theta_d_hist[(write_idx-1) % history_len] = theta_des

    # Pacing (opcional)
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
