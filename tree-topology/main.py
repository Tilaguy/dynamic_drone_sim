import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from collections import deque

# --- TUS CLASES ---
from Dynamic_model import SystemDynamic, MotorDynamics2D
from Control_logic import PositionCascade2D, AttitudeControl2D
from graphic_env import generate_initial_pos

# ================= CONFIGURACIÓN =================
Ts = 0.02          # 50 Hz visual
sub_steps = 40     # Pasos de física por frame
dt = Ts / sub_steps

# --- Configuración Crazyflie 2.1 ---
num_robots = 6
load_mass = 0.020    # 20 gramos de carga máxima recomendada
d = 0.046            # 46 mm

# Dinámica de Motores
max_rpm = 22000.0
kf_rpm = 1.28e-8         # N / RPM^2
km_rpm = 2.0e-10         # Nm / RPM^2
motor_gain = 18.0        # Basado en tau ≈ 0.055s

# Inercia y Masas
robot_masses = np.ones(num_robots) * 0.027 # [Kg]
robot_inertia = [1.4e-5] * num_robots
cables_list = [1.2] * num_robots
cables_list[3:5] = [1., 0.8]
init_load_pos = [0.0, 1.0]

# ================= INICIALIZACIÓN FÍSICA =================
all_masses = np.insert(robot_masses, 0, load_mass)
sys = SystemDynamic(num_robots=num_robots, agent_masses=list(all_masses), robot_inertia=robot_inertia)
sys.Ts = dt
initial_positions = generate_initial_pos(num_robots + 1, cables_list, init_load_pos)
for i, pos in enumerate(initial_positions):
    idx, _ = sys.get_agent_indices(i)
    sys.q[idx : idx + 2, 0] = pos

motors, pos_ctrls, att_ctrls = [], [], []
for i in range(num_robots):
    hover_mass = robot_masses[i] + (load_mass / num_robots)
    mot = MotorDynamics2D(L=d, max_omega_rpm=max_rpm, motor_gain=motor_gain, kf_in_rpm=kf_rpm)
    w_h = mot.set_hover_from_mass(hover_mass)
    print(f"Ω min. required for hovering, per motor: {w_h:.3f} [rad/s]")
    mot.omega[:] = w_h
    motors.append(mot)

    pos_ctrls.append(PositionCascade2D(kp_x=2.0, ki_x=0.01, kd_x=1.5, kp_y=12.0, ki_y=8.0, kd_y=6.0,
                                       ax_limit=2.0, ay_limit=2.0, L=d, kf=mot.kf, omega_hover=w_h))
    att_ctrls.append(AttitudeControl2D(L=d, kf=mot.kf, omega_hover=w_h, kp_theta=0.12, kd_theta=0.03))

# ================= DASHBOARD UI =================
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(4, 2, width_ratios=[1, 1.2]) # 4 filas para incluir la Carga
# 1. Simulación (Izquierda)
ax_sim = fig.add_subplot(gs[:, 0])
ax_sim.set_xlim(-3, 3); ax_sim.set_ylim(0, 5)
ax_sim.set_aspect('equal'); ax_sim.grid(True, alpha=0.3)
line_cables, = ax_sim.plot([], [], 'k-', lw=1, alpha=0.5)
line_drones, = ax_sim.plot([], [], 'b-', lw=3, label='Drones (Inclination)')
pt_load, = ax_sim.plot([], [], 'ro', markersize=10)

# Sliders de Referencia y Selección
ax_sx = plt.axes([0.1, 0.08, 0.3, 0.02])
s_ref_x = Slider(ax_sx, 'Load X', -2.0, 2.0, valinit=0.0)
ax_sy = plt.axes([0.1, 0.05, 0.3, 0.02])
s_ref_y = Slider(ax_sy, 'Load Y', 0.5, 4.0, valinit=1.0)
ax_si = plt.axes([0.1, 0.02, 0.3, 0.02])
s_robot_idx = Slider(ax_si, 'Show Robot', 0, num_robots-1, valinit=0, valstep=1)

# 2. Telemetría (Derecha)
ax_rpm = fig.add_subplot(gs[0, 1])
l_rpm1, = ax_rpm.plot([], [], 'r', label='M1'); l_rpm2, = ax_rpm.plot([], [], 'b', label='M2')
ax_rpm.set_ylabel('RPM'); ax_rpm.legend(loc='upper right', fontsize='x-small'); ax_rpm.set_ylim(0, max_rpm*1.2)

ax_pos = fig.add_subplot(gs[1, 1], sharex=ax_rpm)
l_x_act, = ax_pos.plot([], [], 'b', label='X Robot'); l_z_act, = ax_pos.plot([], [], 'g', label='Y Robot')
ax_pos.set_ylabel('Robot Pos [m]'); ax_pos.legend(loc='upper right', fontsize='x-small'); ax_pos.set_ylim(-4, 6)

ax_phi = fig.add_subplot(gs[2, 1], sharex=ax_rpm)
l_phi_ref, = ax_phi.plot([], [], 'k--', alpha=0.5); l_phi_act, = ax_phi.plot([], [], 'orange', label='Phi')
ax_phi.set_ylabel('Phi [rad]'); ax_phi.legend(loc='upper right', fontsize='x-small'); ax_phi.set_ylim(-1.04, 1.04)

ax_load = fig.add_subplot(gs[3, 1], sharex=ax_rpm)
l_load_x_ref, = ax_load.plot([], [], 'r--', alpha=0.4, label='Ref X')
l_load_x_act, = ax_load.plot([], [], 'r', lw=1.5, label='Load X')
l_load_y_ref, = ax_load.plot([], [], 'g--', alpha=0.4, label='Ref Y')
l_load_y_act, = ax_load.plot([], [], 'g', lw=1.5, label='Load Y')
ax_load.set_ylabel('LOAD Pos [m]'); ax_load.legend(loc='upper right', fontsize='x-small')
ax_load.set_xlabel('Tiempo [s]'); ax_load.set_ylim(-4, 6)

for ax in [ax_rpm, ax_pos, ax_phi, ax_load]: ax.grid(True, linestyle=':', alpha=0.6)
# Buffers
h_len = 250
t_d = deque(maxlen=h_len)
rpm_d = [deque(maxlen=h_len) for _ in range(2)]
pos_d = [deque(maxlen=h_len) for _ in range(2)]
phi_d = [deque(maxlen=h_len) for _ in range(2)]
load_d = {'x_ref': deque(maxlen=h_len), 'x_act': deque(maxlen=h_len),
          'y_ref': deque(maxlen=h_len), 'y_act': deque(maxlen=h_len)}

time_elapsed = 0.0

def update(frame):
    global time_elapsed
    tx, ty = s_ref_x.val, s_ref_y.val
    ridx = int(s_robot_idx.val)

    # Datos para capturar
    temp_data = {}

    for _ in range(sub_steps):
        f_list, t_list = [], []
        for i in range(num_robots):
            idx, _ = sys.get_agent_indices(i + 1)
            x, z, phi = sys.q[idx:idx+3, 0]
            vx, vz, dphi = sys.d_q[idx:idx+3, 0]

            # Referencias relativas (Mantienen formación original pero siguen a la carga)
            rx = initial_positions[i+1][0] + tx
            ry = initial_positions[i+1][1] + (ty - init_load_pos[1])

            _, _, p_des, Dw_F = pos_ctrls[i].step(xd=rx, x=x, vx=vx, yd=ry, y=z, vy=vz,
                                                  m=robot_masses[i] + load_mass/num_robots, dt=dt)
            p_des = np.clip(p_des, -0.6, 0.6)
            Dw_phi = att_ctrls[i].attitude_channels(p_des, phi, dphi)

            w_real, T, _ = motors[i].update(dt, Dw=(np.clip(Dw_F, -0.8*motors[i].omega_h, 0.8*motors[i].omega_h), Dw_phi))
            f_list.append(float(np.sum(T))); t_list.append(d * (T[1] - T[0]))

            if i == ridx:
                temp_data['rpm'] = w_real * 60 / (2*np.pi)
                temp_data['pos'] = [x, z]
                temp_data['phi'] = [p_des, phi]

        sys.step(f_list, t_list)
        sys.update_states()
        time_elapsed += dt

    # Actualizar Históricos
    t_d.append(time_elapsed)
    rpm_d[0].append(temp_data['rpm'][0]); rpm_d[1].append(temp_data['rpm'][1])
    pos_d[0].append(temp_data['pos'][0]); pos_d[1].append(temp_data['pos'][1])
    phi_d[0].append(temp_data['phi'][0]); phi_d[1].append(temp_data['phi'][1])

    load_d['x_ref'].append(tx); load_d['x_act'].append(sys.q[0,0])
    load_d['y_ref'].append(ty); load_d['y_act'].append(sys.q[1,0])

    # Redibujar Telemetría Robot seleccionado
    l_rpm1.set_data(t_d, rpm_d[0]); l_rpm2.set_data(t_d, rpm_d[1])
    l_x_act.set_data(t_d, pos_d[0]); l_z_act.set_data(t_d, pos_d[1])
    l_phi_ref.set_data(t_d, phi_d[0]); l_phi_act.set_data(t_d, phi_d[1])

    # Redibujar Carga (Global)
    l_load_x_ref.set_data(t_d, load_d['x_ref']); l_load_x_act.set_data(t_d, load_d['x_act'])
    l_load_y_ref.set_data(t_d, load_d['y_ref']); l_load_y_act.set_data(t_d, load_d['y_act'])

    # Ejes dinámicos
    ax_rpm.set_xlim(max(0, time_elapsed - 5), time_elapsed + 0.1)
    ax_rpm.set_title(f"Telemetría Robot {ridx}", fontsize=10)

    # Simulación Visual
    q = sys.q
    dx, dy = [], []
    visual_arm = d * 2

    for i in range(1, sys.N):
        idx, _ = sys.get_agent_indices(i)
        x_c, y_c, phi_c = q[idx:idx+3, 0]

        # Calculamos los puntos del "brazo" del dron
        # Usamos cos y sin para rotar la línea según phi
        x1 = x_c - visual_arm * np.cos(phi_c)
        y1 = y_c - visual_arm * np.sin(phi_c)
        x2 = x_c + visual_arm * np.cos(phi_c)
        y2 = y_c + visual_arm * np.sin(phi_c)

        # Agregamos a la lista con None para separar cada dron
        dx.extend([x1, x2, None])
        dy.extend([y1, y2, None])

    line_drones.set_data(dx, dy)

    cx, cy = [], []
    for (p, c) in sys.edges:
        pi, _ = sys.get_agent_indices(p); ci, _ = sys.get_agent_indices(c)
        cx.extend([q[pi,0], q[ci,0], None]); cy.extend([q[pi+1,0], q[ci+1,0], None])
    line_cables.set_data(cx, cy)

    # Dibujar Carga
    pt_load.set_data([q[0, 0]], [q[1, 0]])

    return line_cables, line_drones, pt_load, l_rpm1, l_x_act, l_load_x_act

ani = FuncAnimation(fig, update, frames=None, interval=Ts*1000, blit=False)
plt.show()
