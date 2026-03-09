import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from collections import deque

# --- YOUR CLASSES (Make sure they are in the same folder) ---
from Dynamic_model import SystemDynamic, MotorDynamics2D
from Control_logic import PositionCascade2D, AttitudeControl2D
from graphic_env import generate_initial_position, generate_position_ref

# ================= CRAZYFLIE 2.1 CONFIGURATION =================
Ts = 0.02
sub_steps = 40
dt = Ts / sub_steps

num_robots = 4
load_mass = 0.020    # 20 grams
d = 0.046            # 46 mm (arm length)
max_rpm = 22000.0
kf_rpm = 1.28e-8
km_rpm = 2.0e-10
motor_gain = 18.0

robot_masses = np.full(num_robots, 0.027)
robot_inertia = [1.4e-5] * num_robots
cables_list = [1.2] * num_robots
s = [1.5] * int(math.log2(num_robots + 1))
# s = [d * 1.5] * int(math.log2(num_robots + 1))
init_load_pos = [0.0, 1.0]

# ================= PHYSICAL INITIALIZATION =================
all_masses = np.insert(robot_masses, 0, load_mass)
sys = SystemDynamic(num_robots=num_robots, agent_masses=list(all_masses), robot_inertia=robot_inertia)
print(f"Edges set: {sys.edges}")
sys.Ts = dt
positions_ref = generate_initial_position(N=num_robots, cables_len=cables_list, load_position=init_load_pos, edges=sys.edges)
# positions_ref = generate_position_ref(positions_ref, s=s)

for i, pos in enumerate(positions_ref):
  idx, _ = sys.get_agent_indices(i)
  sys.q[idx : idx + 2, 0] = pos

motors, pos_ctrls, att_ctrls = [], [], []
max_w_h = 0.0
for i in range(num_robots):
  hover_mass = robot_masses[i]
  mot = MotorDynamics2D(L=d, max_omega_rpm=max_rpm, motor_gain=motor_gain, kf_in_rpm=kf_rpm)
  w_h = mot.set_hover_from_mass(hover_mass)

  if max_w_h < w_h:
    max_w_h = w_h

  mot.omega[:] = w_h
  motors.append(mot)

  pos_ctrls.append(PositionCascade2D(
    kp_x=2.3, ki_x=1.5, kd_x=1.5,
    kp_y=12.0, ki_y=8.6, kd_y=6.0,
    ax_limit=2.0, ay_limit=2.0,
    L=d, kf=mot.kf, omega_hover=w_h))

  att_ctrls.append(AttitudeControl2D(
    L=d, kf=mot.kf, omega_hover=w_h,
    kp_theta=0.12, kd_theta=0.03))

max_w_h = max_w_h * 60 / (2*np.pi)
print(f"Maximum ω_h = {max_w_h} [rad/s]")

# ================= DASHBOARD UI =================
fig = plt.figure(figsize=(8, 5))
gs = GridSpec(4, 2, width_ratios=[1, 1.2])

# 1. Simulation View (Left)
ax_sim = fig.add_subplot(gs[:, 0])
ax_sim.set_xlim(-3, 3)
ax_sim.set_ylim(0, 5)
ax_sim.set_aspect('equal')
ax_sim.grid(True, alpha=0.3)

line_cables, = ax_sim.plot([], [], 'k-', lw=1, alpha=0.5)
line_drones, = ax_sim.plot([], [], 'b-', lw=3, label='Drones')
pt_load, = ax_sim.plot([], [], 'ro', markersize=10, zorder=5)

# --- SLIDERS ---
ax_sx = plt.axes([0.1, 0.11, 0.3, 0.02]) # type: ignore
s_ref_x = Slider(ax_sx, 'Load X Ref', -2.0, 2.0, valinit=0.0)

ax_sy = plt.axes([0.1, 0.08, 0.3, 0.02]) # type: ignore
s_ref_y = Slider(ax_sy, 'Load Y Ref', 0.5, 4.0, valinit=1.0)

ax_si = plt.axes([0.1, 0.05, 0.3, 0.02]) # type: ignore
s_robot_idx = Slider(ax_si, 'Show Robot', 1, num_robots, valinit=0, valstep=1)

ax_sp = plt.axes([0.1, 0.02, 0.3, 0.02], facecolor='mistyrose') # type: ignore
s_perturb = Slider(ax_sp, 'PERTURB X (N)', -1.5, 1.5, valinit=0.0)

# 2. Telemetry (Right)
ax_rpm = fig.add_subplot(gs[0, 1])
l_rpm1, = ax_rpm.plot([], [], 'r', label='M1')
l_rpm2, = ax_rpm.plot([], [], 'b', label='M2')
l_rpm_ref, = ax_rpm.plot([], [], '--k', label='Hover ref', alpha=0.4)
ax_rpm.set_ylabel('RPM')
ax_rpm.legend(loc='upper right', fontsize='x-small')
ax_rpm.set_ylim(max_w_h * 0.8, max_w_h * 1.2)

ax_pos = fig.add_subplot(gs[1, 1], sharex=ax_rpm)
l_x_act, = ax_pos.plot([], [], 'b', label='Robot X')
l_x_ref, = ax_pos.plot([], [], '--b', label='X ref', alpha=0.4)
l_z_act, = ax_pos.plot([], [], 'g', label='Robot Y')
l_z_ref, = ax_pos.plot([], [], '--g', label='Y ref', alpha=0.4)
ax_pos.set_ylabel('Robot Pos [m]')
ax_pos.legend(loc='upper right', fontsize='x-small')
ax_pos.set_ylim(-4, 6)

ax_phi = fig.add_subplot(gs[2, 1], sharex=ax_rpm)
l_phi_ref, = ax_phi.plot([], [], 'k--', alpha=0.4)
l_phi_act, = ax_phi.plot([], [], 'orange', label='Phi')
ax_phi.set_ylabel('Phi [rad]')
ax_phi.legend(loc='upper right', fontsize='x-small')
ax_phi.set_ylim(-1.04, 1.04)

ax_load = fig.add_subplot(gs[3, 1], sharex=ax_rpm)
l_load_x_ref, = ax_load.plot([], [], 'r--', alpha=0.4, label='Ref X')
l_load_x_act, = ax_load.plot([], [], 'r', lw=1.5, label='Load X')
l_load_y_ref, = ax_load.plot([], [], 'g--', alpha=0.4, label='Ref Y')
l_load_y_act, = ax_load.plot([], [], 'g', lw=1.5, label='Load Y')
ax_load.set_ylabel('Load Pos [m]')
ax_load.legend(loc='upper right', fontsize='x-small')
ax_load.set_xlabel('Time [s]')
ax_load.set_ylim(-4, 6)

for ax in [ax_rpm, ax_pos, ax_phi, ax_load]:
  ax.grid(True, linestyle=':', alpha=0.6)

# Buffers
history_len = 250
t_d = deque(maxlen=history_len)
rpm_d = [deque(maxlen=history_len) for _ in range(2)]
rpm_ref_d = deque(maxlen=history_len)
pos_d = [deque(maxlen=history_len) for _ in range(2)]
pos_ref = [deque(maxlen=history_len) for _ in range(2)]
phi_d = [deque(maxlen=history_len) for _ in range(2)]

load_d = {
  'x_ref': deque(maxlen=history_len),
  'x_act': deque(maxlen=history_len),
  'y_ref': deque(maxlen=history_len),
  'y_act': deque(maxlen=history_len)
}

time_elapsed = 0.0
parent_of  = {child: parent for parent, child in sys.edges}

def update(frame):
  global time_elapsed, positions_ref
  positions_ref = generate_position_ref(positions_ref, s=s)

  tx, ty = s_ref_x.val, s_ref_y.val
  ridx = int(s_robot_idx.val)
  f_perturb = s_perturb.val  # Capture impulse value

  temp_data = {}

  for _ in range(sub_steps):

    force_list, torque_list = [], []
    L_list = np.zeros(num_robots)

    for i in range(num_robots):

      idx, _ = sys.get_agent_indices(i + 1)

      x, y, phi = sys.q[idx:idx+3, 0]
      vx, vz, dphi = sys.d_q[idx:idx+3, 0]

      rx = positions_ref[i+1][0] + tx
      ry = positions_ref[i+1][1] + (ty - init_load_pos[1])

      for _ in range(sub_steps):
        _, _, p_des, Dw_F = pos_ctrls[i].step(
          xd=rx, x=x, vx=vx,
          yd=ry, y=y, vy=vz,
          m=robot_masses[i],
          dt=dt/sub_steps)

        p_des = np.clip(p_des, -0.6, 0.6)

        Dw_phi = att_ctrls[i].attitude_channels(p_des, phi, dphi)

      w_real, F, _ = motors[i].update(
        dt,
        Dw=(np.clip(Dw_F, -0.8*motors[i].omega_h, 0.8*motors[i].omega_h), Dw_phi)
      )

      force_list.append(float(np.sum(F)))
      torque_list.append(d * (F[1] - F[0]))

      i_rho = parent_of[i + 1]
      idx_rho, _ = sys.get_agent_indices(i_rho)
      x_rho, y_rho = sys.q[idx_rho:idx_rho+2, 0]
      L_list[i] = np.sqrt((x - x_rho) ** 2 + (y - y_rho) ** 2)
      if i == (ridx - 1):
        temp_data['rpm'] = w_real * 60 / (2*np.pi)
        temp_data['rpm_h'] = motors[i].omega_h * 60 / (2*np.pi)
        temp_data['pos'] = [x, y]
        temp_data["pos_ref"] = [rx, ry]
        temp_data['phi'] = [p_des, phi]

    _, lambdas = sys.step(force_list, torque_list, f_perturb)
    T = 2 * lambdas.flatten() * np.array(cables_list)
    print(f"\r|T| = {np.abs(T)} [N], L = {L_list} [m]", end="", flush=True)

    time_elapsed += dt

    # SLIDER RESET: makes it behave like an impulse
    if f_perturb != 0:
      s_perturb.set_val(0.0)

  # Update history buffers
  t_d.append(time_elapsed)
  rpm_d[0].append(temp_data['rpm'][0])
  rpm_d[1].append(temp_data['rpm'][1])
  rpm_ref_d.append(temp_data['rpm_h'])
  pos_d[0].append(temp_data['pos'][0])
  pos_d[1].append(temp_data['pos'][1])
  pos_ref[0].append(temp_data['pos_ref'][0])
  pos_ref[1].append(temp_data['pos_ref'][1])
  phi_d[0].append(temp_data['phi'][0])
  phi_d[1].append(temp_data['phi'][1])

  load_d['x_ref'].append(tx)
  load_d['x_act'].append(sys.q[0,0])
  load_d['y_ref'].append(ty)
  load_d['y_act'].append(sys.q[1,0])

  # Redraw Telemetry
  l_rpm1.set_data(t_d, rpm_d[0]); l_rpm2.set_data(t_d, rpm_d[1])
  l_rpm_ref.set_data(t_d, rpm_ref_d)
  l_x_act.set_data(t_d, pos_d[0]); l_z_act.set_data(t_d, pos_d[1])
  l_x_ref.set_data(t_d, pos_ref[0]); l_z_ref.set_data(t_d, pos_ref[1])
  l_phi_ref.set_data(t_d, phi_d[0]); l_phi_act.set_data(t_d, phi_d[1])
  l_load_x_ref.set_data(t_d, load_d['x_ref']); l_load_x_act.set_data(t_d, load_d['x_act'])
  l_load_y_ref.set_data(t_d, load_d['y_ref']); l_load_y_act.set_data(t_d, load_d['y_act'])
  ax_rpm.set_xlim(max(0, time_elapsed - 5), time_elapsed + 0.1)

  # Visual Simulation (Drones as lines)
  q = sys.q
  dx, dy = [], []
  visual_arm = d * 2
  for i in range(1, sys.N + 1):
    idx, _ = sys.get_agent_indices(i)
    xc, yc, phic = q[idx:idx+3, 0]
    x1, y1 = xc - visual_arm * np.cos(phic), yc - visual_arm * np.sin(phic)
    x2, y2 = xc + visual_arm * np.cos(phic), yc + visual_arm * np.sin(phic)
    dx.extend([x1, x2, None]); dy.extend([y1, y2, None])

  line_drones.set_data(dx, dy)
  cx, cy = [], []
  for (p, c) in sys.edges:
    pi, _ = sys.get_agent_indices(p); ci, _ = sys.get_agent_indices(c)
    cx.extend([q[pi,0], q[ci,0], None]); cy.extend([q[pi+1,0], q[ci+1,0], None])
  line_cables.set_data(cx, cy)
  pt_load.set_data([q[0, 0]], [q[1, 0]])

  return line_cables, line_drones, pt_load, l_rpm1, l_x_act, l_load_x_act

ani = FuncAnimation(fig, update, frames=None, interval=Ts*1000, blit=False, cache_frame_data=False)
plt.show()
