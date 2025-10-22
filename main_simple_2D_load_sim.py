# ========= 2D Drone: Attitude + Motors + Rigid Body (Self-contained) =========
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

from sim_2D_load.dynamic_models import RigidBody2D, MotorDynamics
from sim_2D.control import PositionCascade2D, AttitudeControl2D

# ============================= Simulation Setup =============================
# Physical / model params
G = 9.81

drone_mass=0.8
L=0.8
Izz=5e-3
d=1
load_mass=0.2

max_omega_rpm = 20000.0
kf_rpm = 6.11e-8
km_rpm = 1.5e-9

USE_GPU = False

dt = 0.01
sim_time = 10.0
steps = int(sim_time / dt)
sub_steps = 2
sub_dt = dt / sub_steps

# Desired references (sliders will override during run)
desired_x   = 0.0
desired_y   = 2.0
desired_phi  = 0.0  # rad
desired_th  = 0.0  # rad
φ_des_prev = 0.0

# Variables globales para el impulso
PENDING_TH_IMP = 0.0
__resetting_th_slider = False

# Instantiate blocks
drone  = RigidBody2D(drone_mass=drone_mass, L=L, Izz=Izz,
                     d=d, load_mass=load_mass,
                     use_gpu=USE_GPU)
motors = MotorDynamics(kf_in_rpm=kf_rpm, km_in_rpm=km_rpm,
                         motor_gain=15.0, max_omega_rpm=max_omega_rpm, use_gpu=USE_GPU)
# ω_h = motors.set_hover_from_mass(mass)
ω_h = 937.79
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
phi_hist = np.zeros(history_len)
theta_hist = np.zeros(history_len)
phi_d_hist = np.zeros(history_len)
T_hist = np.zeros(history_len)
write_idx = 0

def to_cpu(arr):
    try:
        return arr.get()
    except Exception:
        return arr

# ============================= Plot/Animation =============================
# --- LAYOUT CON GRIDSPEC ---
fig = plt.figure(figsize=(12, 8))

gs = fig.add_gridspec(
    nrows=3, ncols=3,
    height_ratios=[1.0, 1.0, 0.35],   # 2 filas para plots + 1 fila bajita para sliders
    width_ratios=[1.2, 1.0, 1.0],     # columna izquierda un poco más ancha
    left=0.06, right=0.98, bottom=0.06, top=0.95, wspace=0.35, hspace=0.35
)

# Columna izquierda: sim 2D ocupando 2 filas
ax_drone = fig.add_subplot(gs[0:2, 0])

# Fila 1 (arriba), columnas 2 y 3
ax_mot = fig.add_subplot(gs[0, 1])   # Motor speeds
ax_wf  = fig.add_subplot(gs[0, 2])   # Altitude channel Dω_F

# Fila 2 (abajo), columnas 2 y 3
ax_phi     = fig.add_subplot(gs[1, 1])  # φ vs φ_des (y θ)
ax_tension = fig.add_subplot(gs[1, 2])  # Magnitud de tensión (nuevo eje)

# Fila 3: sliders ocupando todo el ancho
ax_sliders = fig.add_subplot(gs[2, :])
ax_sliders.axis("off")

# --- CONFIG DE CADA AXIS (igual que tenías, solo cambiando las referencias) ---

# (izquierda) Drone 2D
ax_drone.set_xlim(-5, 5)
ax_drone.set_ylim(0.0, 6)
ax_drone.set_xlabel("x [m]")
ax_drone.set_ylabel("y [m]")
ax_drone.set_title("2D Drone (x-y)")
ax_drone.set_aspect('equal', adjustable='box')
drone_body, = ax_drone.plot([], [], "o-", lw=3, markersize=5)
load_body,  = ax_drone.plot([], [], "o-", lw=2, markersize=8)

# (0,1) Motor speeds
ax_mot.set_title("Motor speeds ω_i(t)")
ax_mot.set_xlim(0, history_len)
ax_mot.set_ylim(0, 1.1 * (max_omega_rpm * 2.0 * math.pi) / 60.0)
ax_mot.set_xlabel("step")
ax_mot.set_ylabel("ω [rad/s]")
lines_motors = [ax_mot.plot([], [])[0] for _ in range(2)]
ax_mot.legend([f"M{i+1}" for i in range(2)], loc="upper right")

# (0,2) Dω_F (altitud)
ax_wf.set_title("Altitude channel Dω_F")
ax_wf.set_xlim(0, history_len)
ax_wf.set_ylim(-800.0, 800.0)
ax_wf.set_xlabel("step")
ax_wf.set_ylabel("Dω_F [rad/s]")
line_wf, = ax_wf.plot([], [])

# (1,1) φ, φ_des, θ
ax_phi.set_title("Attitude: φ, φ_des and θ")
ax_phi.set_xlim(0, history_len)
ax_phi.set_ylim(-0.6, 0.6)
ax_phi.set_xlabel("step")
ax_phi.set_ylabel("angle [rad]")
line_phi,   = ax_phi.plot([], [])
line_phi_d, = ax_phi.plot([], [])
line_theta, = ax_phi.plot([], [])
ax_phi.legend(["φ", "φ_des", "θ"], loc="upper right")

# (1,2) Magnitud de tensión (usa tu señal real aquí)
ax_tension.set_title("Magnitude of tension")
ax_tension.set_xlim(0, history_len)
ax_tension.set_ylim(-load_mass*G*1.5, load_mass*G*1.5 )
ax_tension.set_xlabel("step")
ax_tension.set_ylabel("|T| [N]")
line_tension, = ax_tension.plot([], [])

# --- Inicialización de líneas ---
y_blank = np.full(history_len, np.nan)
for i in range(2):
    lines_motors[i].set_data(time_hist, y_blank.copy())
line_wf.set_data(time_hist, y_blank.copy())
line_phi.set_data(time_hist, y_blank.copy())
line_theta.set_data(time_hist, y_blank.copy())
line_phi_d.set_data(time_hist, y_blank.copy())
line_tension.set_data(time_hist, y_blank.copy())

def th_impulse_cb(val):
    global PENDING_TH_IMP, __resetting_th_slider
    if __resetting_th_slider:
        return
    # Si el usuario movió el slider lejos de 0, lo tratamos como un impulso
    if abs(val) > 1e-9:
        PENDING_TH_IMP += float(val)  # acumulamos por si hace varios “clicks” seguidos
        # Auto-reset del slider a 0 (sin re-disparar)
        __resetting_th_slider = True
        s_th.set_val(0.0)
        __resetting_th_slider = False

# --- Sliders dentro del panel inferior (posiciones relativas al bbox del panel sliders) ---
bbox = ax_sliders.get_position()  # en coords de figura
x0, y0, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height

slot_h   = h * 0.22   # alto de cada slider
v_gap    = h * 0.06   # separación vertical
x_pad    = w * 0.05
slider_w = w * 0.90

# fila superior de sliders (de arriba hacia abajo)
y_top = y0 + h - slot_h - v_gap*0.5

ax_sx   = fig.add_axes([x0 + x_pad, y_top - 0*(slot_h + v_gap), slider_w, slot_h]) # type: ignore
ax_sy   = fig.add_axes([x0 + x_pad, y_top - 1*(slot_h + v_gap), slider_w, slot_h]) # type: ignore
ax_sphi = fig.add_axes([x0 + x_pad, y_top - 2*(slot_h + v_gap), slider_w, slot_h]) # type: ignore
ax_sth  = fig.add_axes([x0 + x_pad, y_top - 3*(slot_h + v_gap), slider_w, slot_h]) # type: ignore

s_x   = Slider(ax_sx,   "x_d [m]",        -4, 4,    valinit=desired_x)
s_y   = Slider(ax_sy,   "y_d [m]",         0.0, 5.8, valinit=desired_y)
s_phi = Slider(ax_sphi, "φ_d [rad]",      -0.3, 0.3, valinit=desired_phi)
s_th  = Slider(ax_sth,  "Impulso θ [rad]", -1.05, 1.05, valinit=0.0)

s_x.on_changed(lambda val: globals().__setitem__('desired_x', s_x.val))
s_y.on_changed(lambda val: globals().__setitem__('desired_y', s_y.val))
s_phi.on_changed(lambda val: globals().__setitem__('desired_phi', s_phi.val))
s_th.on_changed(th_impulse_cb)

def plots_update(state, W_i, Dω_F):
    global write_idx
    # Drone body segment in world (x-y)
    φ = float(state["drone"]["angle"])
    x = float(state["drone"]["position"][0])
    y = float(state["drone"]["position"][1])
    θ = float(state["load"]["angle"])
    x_l = float(state["load"]["position"][0])
    y_l = float(state["load"]["position"][1])
    x1 = x - L * math.cos(φ)
    y1 = y - L * math.sin(φ)
    x2 = x + L * math.cos(φ)
    y2 = y + L * math.sin(φ)
    drone_body.set_data([x1, x2], [y1, y2])
    load_body.set_data([x, x_l], [y, y_l])


    ax = float(state["drone"]["lin_acc"][0])
    ay = float(state["drone"]["lin_acc"][1])
    c_th = math.cos(θ)
    s_th = math.sin(θ)
    T = load_mass * (d * θ**2 + ax * s_th + (G - ay) * c_th)

    # Histories
    for i in range(2):
        omega_hist[i][write_idx] = float(to_cpu(W_i[i]))
        lines_motors[i].set_ydata(omega_hist[i])

    wf_hist[write_idx] = float(Dω_F)
    line_wf.set_ydata(wf_hist)

    phi_hist[write_idx]   = φ
    theta_hist[write_idx]   = θ
    phi_d_hist[write_idx] = float(desired_phi)
    T_hist[write_idx] = float(T)
    line_phi.set_ydata(phi_hist)
    line_phi_d.set_ydata(phi_d_hist)
    line_theta.set_ydata(theta_hist)
    line_tension.set_ydata(T_hist)

    write_idx = (write_idx + 1) % history_len

last_time = time.perf_counter()

def update(frame):
    global last_time, φ_des_prev, PENDING_TH_IMP

    # Estado actual
    state = drone.state()
    x,  y  = state["drone"]["position"]
    vx, vy = state["drone"]["velocity"]
    φ,  ω  = state["drone"]["angle"], state["drone"]["omega"]

    if abs(PENDING_TH_IMP) > 0.0:
        try:
            # Impulso como Δθ (puedes cambiar a domega si te interesa un “golpe” de velocidad)
            drone.impulse_on_load(dtheta=PENDING_TH_IMP, domega=0.0)
        finally:
            PENDING_TH_IMP = 0.0

    ax_ref, ay_ref, φ_des_raw, Dω_F = pos.step(xd=desired_x, x=x, vx=vx,
                                               yd=desired_y, y=y, vy=vy,
                                               m=drone_mass, dt=dt)

    max_dphi = 2.0  # [rad/s] velocidad máx. de referencia
    dφ_cmd = np.clip(φ_des_raw - φ_des_prev, -max_dphi*dt, max_dphi*dt)
    phi_des = np.clip(φ_des_prev + dφ_cmd, -0.5, 0.5)
    φ_des_prev = float(phi_des)

    # Lazo de actitud -> canal diferencial
    Dω_φ = att.attitude_channels(phi_des, φ, ω)

    # Saturaciones de canales (protege ω_des)
    Dω_F = np.clip(Dω_F, -0.5*ω_h, 0.5*ω_h)   # p.ej. ±50% de ω_h
    Dω_φ = np.clip(Dω_φ, -0.5*ω_h, 0.5*ω_h)

    # Subpasos para estabilidad numérica
    for _ in range(sub_steps):
        W_i, T_i, M_i = motors.update(sub_dt, Dw=(Dω_F + ω_h, Dω_φ))
        state = drone.update(sub_dt, T_i)

    # Plots
    if frame % 1 == 0:
        plots_update(state, W_i, Dω_F)
        # (opcional) refleja la ref. usada en el gráfico de φ_d:
        phi_d_hist[(write_idx-1) % history_len] = phi_des

    # Pacing (opcional)
    elapsed = time.perf_counter() - last_time
    last_time = time.perf_counter()
    sleep_time = dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

    return drone_body, load_body, *lines_motors, line_wf, line_phi, line_phi_d, line_theta, line_tension

def init():
    return (drone_body, load_body, *lines_motors, line_wf, line_phi, line_phi_d, line_theta, line_tension)

ani = animation.FuncAnimation(fig, update, frames=steps, blit=True, interval=dt*1000, init_func=init)
for ln in [drone_body, load_body, *lines_motors, line_wf, line_phi, line_phi_d, line_theta, line_tension]:
    ln.set_animated(True)
for i in range(2):
    lines_motors[i].set_xdata(time_hist)
line_wf.set_xdata(time_hist)
line_phi.set_xdata(time_hist)
line_phi_d.set_xdata(time_hist)
line_theta.set_xdata(time_hist)
line_tension.set_xdata(time_hist)

plt.show()
