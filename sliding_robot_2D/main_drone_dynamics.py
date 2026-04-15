import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from collections import deque

from Dynamic_model import PulleyDynamic_Lagrange2D, DroneDynamic2D, MotorDynamics2D
from Control_logic import PositionCascade2D, AttitudeControl2D
from Geometrical_functios import FormationManager

Ts = 0.02
sub_steps = 40
dt = Ts / sub_steps

num_robots = 2
load_mass = 0.0027

# ================= CRAZYFLIE 2.1 CONFIGURATION =================
d = 0.046
max_rpm = 22000.0
kf_rpm = 1.28e-8
km_rpm = 2.0e-10
motor_gain = 18.0
robot_mass = 0.027
robot_inertia = 1.4e-5

robot_masses = np.full(num_robots, robot_mass)
init_load_pos = [0.0, 1.0]
init_cable_len = 4
# s = 7 * d
# print(f"safe distance: {s} m")

# ================= PHYSICAL INITIALIZATION =================
pulley = PulleyDynamic_Lagrange2D(mass=load_mass, pulley_pos=np.array([0]), cable_len=init_cable_len, Ts=dt, damping=0.025)
edges=[(0, 1), (0, 2)]
print(f"Edges set: {edges}")

pos_manager = FormationManager(N=num_robots, edges=[(0, 1), (0, 2)])
cables_list = pos_manager.exponential_cables(base_length=init_cable_len/2., alpha=1)
print(f"Cables length: {cables_list}")
positions_ref = pos_manager.generate_initial_position(load_position=init_load_pos, base_sep=init_cable_len*0.9)
print(positions_ref)

motors, pos_ctrls, att_ctrls, bases = [], [], [], []
max_w_h = 0.0
for i in range(num_robots):
	base = DroneDynamic2D(mass=robot_mass, inertia=robot_inertia, ini_pos=positions_ref[i + 1, :], Ts=dt)

	bases.append(base)

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
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(4, 2, width_ratios=[1, 1.2])

# 1. Simulation View (Left)
ax_sim = fig.add_subplot(gs[:, 0])
ax_sim.set_xlim(-3, 3)
ax_sim.set_ylim(0, 5)
ax_sim.set_aspect('equal')
ax_sim.grid(True, alpha=0.3)

text_telemetry = ax_sim.text(0.05, 0.95, '', transform=ax_sim.transAxes,
														 ha='left', va='top', fontsize=10, family='monospace',
														 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

cmap_cables = mcolors.LinearSegmentedColormap.from_list('tension', ['black', 'red'])
cable_collection = LineCollection([], cmap=cmap_cables, lw=1, alpha=0.8)
ax_sim.add_collection(cable_collection)

line_drones, = ax_sim.plot([], [], 'b-', lw=2, label='Drones')
pt_load, = ax_sim.plot([], [], 'ro', markersize=10, zorder=5)

# --- SLIDERS ---
_, x1, x2 = positions_ref
ax1_sx = plt.axes([0.1, 0.17, 0.3, 0.02]) # type: ignore
s_ref_x1 = Slider(ax1_sx, 'D1 X Ref', -2.0, 2.0, valinit=x1[0])

ax1_sy = plt.axes([0.1, 0.14, 0.3, 0.02]) # type: ignore
s_ref_y1 = Slider(ax1_sy, 'D1 Y Ref', 0.0, 4.0, valinit=x1[1])

ax2_sx = plt.axes([0.1, 0.11, 0.3, 0.02]) # type: ignore
s_ref_x2 = Slider(ax2_sx, 'D2 X Ref', -2.0, 2.0, valinit=x2[0])

ax2_sy = plt.axes([0.1, 0.08, 0.3, 0.02]) # type: ignore
s_ref_y2 = Slider(ax2_sy, 'D2 Y Ref', 0.0, 4.0, valinit=x2[1])

ax_si = plt.axes([0.1, 0.05, 0.3, 0.02]) # type: ignore
s_robot_idx = Slider(ax_si, 'Show Robot', 1, num_robots, valinit=1, valstep=1)

ax_sp = plt.axes([0.1, 0.02, 0.3, 0.02], facecolor='mistyrose') # type: ignore
s_perturb = Slider(ax_sp, 'PERTURB X (N)', -8, 8, valinit=0.0)

# Telemetry (Right)
max_omega = (max_rpm * 2.0 * math.pi) / 60.0
ax_rpm = fig.add_subplot(gs[0, 1])
l_rpm1, = ax_rpm.plot([], [], 'r', label='M1')
l_rpm2, = ax_rpm.plot([], [], 'b', label='M2')
l_rpm_ref, = ax_rpm.plot([], [], '--k', label='Hover ref', alpha=0.4)
ax_rpm.set_ylabel('RPM')
ax_rpm.legend(loc='upper right', fontsize='x-small')
ax_rpm.set_ylim(max_w_h * 0.8, max(max_omega * 1.2, max_w_h * 1.2))

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

def update(frame):
	global time_elapsed, positions_ref

	p_ref = np.array([[s_ref_x1.val, s_ref_y1.val],
		[s_ref_x2.val, s_ref_y2.val]])
	ridx = int(s_robot_idx.val) - 1
	f_dist = s_perturb.val  # Capture impulse value

	temp_data = {}
	force_list, torque_list = [], []

	for step_idx in range(sub_steps):
	 	# 1. COMPUTE CONTROLS & FORCES FOR EACH DRONE
		for i in range(num_robots):
			_, _, phi_des, Dw_F = pos_ctrls[i].step(
				xd=p_ref[i, 0], x=bases[i].q[0, 0], vx=bases[i].dq[0, 0],
				yd=p_ref[i, 1], y=bases[i].q[1, 0], vy=bases[i].dq[1, 0],
				m=bases[i].m,
				dt=dt)

			phi_des = np.clip(phi_des, -0.6, 0.6)
			Dw_phi = att_ctrls[i].attitude_channels(phi_des, bases[i].q[2, 0], bases[i].dq[2, 0])

			w_real, F, _ = motors[i].update(
				dt,
				Dw=(np.clip(Dw_F, -0.8*motors[i].omega_h, 0.8*motors[i].omega_h), Dw_phi)
			)

			Ft = float(np.sum(F))
			τ = d * (F[1] - F[0])

			F_w = bases[i]._R(bases[i].q[2, 0]) @ np.array([[0.0], [Ft]])
			bases[i].F_t = F_w
			bases[i].τ = τ

			force_list.append(Ft)
			torque_list.append(τ)

		# 2. STEP THE ENTIRE COUPLED PHYSICS SYSTEM ONCE
		pulley.step(base1=bases[0], base2=bases[1], F_ext=f_dist)

		# 3. SYNCHRONIZE THE DRONES TO THE NEW PHYSICS STATES
		for i in range(num_robots):
			bases[i].q[:, 0] = pulley.q[3*i + 2: 3*i + 5, 0]
			bases[i].dq[:, 0] = pulley.dq[3*i + 2: 3*i + 5, 0]
			bases[i].ddq[:, 0] = pulley.ddq[3*i + 2: 3*i + 5, 0]

			# Telemetry logic for the final substep remains exactly the same
			if step_idx == sub_steps - 1:
				if i == ridx:
					temp_data['rpm'] = w_real * 60 / (2*np.pi)
					temp_data['rpm_h'] = motors[i].omega_h * 60 / (2*np.pi)
					temp_data['pos'] = bases[i].q[:2, 0].copy()
					temp_data["pos_ref"] = p_ref[i, :]
					temp_data['phi'] = [phi_des, bases[i].q[2, 0].copy()]

		time_elapsed += dt

	stats_text = (f"► ROBOT {ridx} STATS\n"
               f"Length: {pulley.l[ridx]:.3f} [m]\n"
               f"Total Length: {sum(pulley.l):.3f} [m]\n")

	text_telemetry.set_text(stats_text)

	# SLIDER RESET: makes it behave like an impulse
	if f_dist != 0:
		s_perturb.set_val(0.0)

	# Update history buffers
	t_d.append(time_elapsed)
	rpm_d[0].append(temp_data['rpm'][0]) # type: ignore
	rpm_d[1].append(temp_data['rpm'][1]) # type: ignore
	rpm_ref_d.append(temp_data['rpm_h'])
	pos_d[0].append(temp_data['pos'][0])
	pos_d[1].append(temp_data['pos'][1])
	pos_ref[0].append(temp_data['pos_ref'][0])
	pos_ref[1].append(temp_data['pos_ref'][1])
	phi_d[0].append(temp_data['phi'][0])
	phi_d[1].append(temp_data['phi'][1])

	load_d['x_ref'].append(pulley.q[0,0])
	load_d['x_act'].append(pulley.q[0,0])
	load_d['y_ref'].append(pulley.q[1,0])
	load_d['y_act'].append(pulley.q[1,0])

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
	dx, dy = [], []
	visual_arm = d * 2
	for base in bases:
		xc, yc, phic = base.q[:, 0]
		x1, y1 = xc - visual_arm * np.cos(phic), yc - visual_arm * np.sin(phic)
		x2, y2 = xc + visual_arm * np.cos(phic), yc + visual_arm * np.sin(phic)
		dx.extend([x1, x2, None]); dy.extend([y1, y2, None])

	line_drones.set_data(dx, dy)

	segments = []
	for (_, c) in edges:
		x, y = pulley.q[:2,0]
		x_d, y_d, _ = bases[c - 1].q[:, 0]
		segments.append([(x_d, y_d), (x, y)])

	cable_collection.set_segments(segments)

	T_abs = np.abs(pulley.T) # type: ignore
	t_min = T_abs.min()
	t_max = max(T_abs.max(), 1e-3)

	cable_collection.set_clim(t_min, t_max)
	cable_collection.set_array(T_abs)

	pt_load.set_data(pulley.q[:2])

	return cable_collection, line_drones, pt_load, l_rpm1, l_x_act, l_load_x_act, text_telemetry

ani = FuncAnimation(fig, update, frames=None, interval=Ts*1000, blit=False, cache_frame_data=False)
plt.show()
