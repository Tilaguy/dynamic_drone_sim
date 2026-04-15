import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from collections import deque

from Dynamic_model import PulleyDynamic2D, PulleyDynamic_Lagrange2D
from Geometrical_functios import FormationManager

Ts = 0.02
sub_steps = 40
dt = Ts / sub_steps

num_robots = 2
load_mass = 0.027

init_load_pos = [0.0, 1.0]
init_cable_len = 4

# ================= PHYSICAL INITIALIZATION =================
pulley = PulleyDynamic2D(mass=load_mass, pulley_pos=np.array(init_load_pos), cable_len=init_cable_len, Ts=dt, damping=0.025)
pulley2 = PulleyDynamic_Lagrange2D(mass=load_mass, pulley_pos=np.array(init_load_pos), cable_len=init_cable_len, Ts=dt, damping=0.025)
edges=[(0, 1), (0, 2)]
print(f"Edges set: {edges}")

pos_manager = FormationManager(N=num_robots, edges=[(0, 1), (0, 2)])
cables_list = pos_manager.exponential_cables(base_length=init_cable_len/2., alpha=1)
print(f"Cables length: {cables_list}")
positions_ref = pos_manager.generate_initial_position(load_position=init_load_pos, base_sep=init_cable_len*0.9)
# pulley.x = np.array(positions_ref[1]).reshape(2,1)

# ================= DASHBOARD UI =================
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 2, width_ratios=[1, 1.2])

# 1. Simulation View (Left)
ax_sim = fig.add_subplot(gs[:, 0])
ax_sim.set_xlim(-5, 5)
ax_sim.set_ylim(0, 10)
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
ax1_sx = plt.axes([0.1, 0.13, 0.3, 0.02]) # type: ignore
s_ref_x1 = Slider(ax1_sx, 'D1 X Ref', -5.0, 5.0, valinit=x1[0])

ax1_sy = plt.axes([0.1, 0.105, 0.3, 0.02]) # type: ignore
s_ref_y1 = Slider(ax1_sy, 'D1 Y Ref', 0.0, 10.0, valinit=x1[1])

ax2_sx = plt.axes([0.1, 0.08, 0.3, 0.02]) # type: ignore
s_ref_x2 = Slider(ax2_sx, 'D2 X Ref', -5.0, 5.0, valinit=x2[0])

ax2_sy = plt.axes([0.1, 0.05, 0.3, 0.02]) # type: ignore
s_ref_y2 = Slider(ax2_sy, 'D2 Y Ref', 0.0, 10.0, valinit=x2[1])

ax_sp = plt.axes([0.1, 0.02, 0.3, 0.02], facecolor='mistyrose') # type: ignore
s_perturb = Slider(ax_sp, 'DISTURB X (N)', -8, 8, valinit=0.0)

# Telemetry (Right)
ax_load = fig.add_subplot(gs[0, 1])
l_load_x_ref, = ax_load.plot([], [], 'r--', alpha=0.4, label='Ref X')
l_load_x_act, = ax_load.plot([], [], 'r', lw=1.5, label='Load X')
l_load_y_ref, = ax_load.plot([], [], 'g--', alpha=0.4, label='Ref Y')
l_load_y_act, = ax_load.plot([], [], 'g', lw=1.5, label='Load Y')
ax_load.set_ylabel('Load Pos [m]')
ax_load.legend(loc='upper right', fontsize='x-small')
ax_load.set_xlabel('Time [s]')
ax_load.set_ylim(-4, 6)
ax_load.grid(True, linestyle=':', alpha=0.6)

# Buffers
history_len = 250
t_d = deque(maxlen=history_len)

load_d = {
  'x_ref': deque(maxlen=history_len),
  'x_act': deque(maxlen=history_len),
  'y_ref': deque(maxlen=history_len),
  'y_act': deque(maxlen=history_len)
}

time_elapsed = 0.0

def update(frame):
  global time_elapsed, positions_ref

  t1 = np.array([s_ref_x1.val, s_ref_y1.val])
  t2 = np.array([s_ref_x2.val, s_ref_y2.val])
  f_perturb = s_perturb.val  # Capture impulse value

  positions_ref = [positions_ref[0], t1, t2]
  _, x1, x2 = positions_ref
  for _ in range(sub_steps):
    pulley.step(x1=x1.reshape((2,1)), x2=x2.reshape((2,1)), F_ext=f_perturb)

    time_elapsed += dt

  stats_text = (f"► STATS\n"
                f"δl_1: {pulley.l[0]:.3f} [m]\n"
                f"δl_2: {pulley.l[1]:.3f} [m]\n"
                f"LENGTH: {sum(pulley.l):.3f} [m]")

  text_telemetry.set_text(stats_text)

  # SLIDER RESET: makes it behave like an impulse
  if f_perturb != 0:
    s_perturb.set_val(0.0)

  # Update history buffers
  t_d.append(time_elapsed)

  load_d['x_ref'].append(pulley.x[0,0])
  load_d['x_act'].append(pulley.x[0,0])
  load_d['y_ref'].append(pulley.x[1,0])
  load_d['y_act'].append(pulley.x[1,0])

  # Redraw Telemetry
  l_load_x_ref.set_data(t_d, load_d['x_ref']); l_load_x_act.set_data(t_d, load_d['x_act'])
  l_load_y_ref.set_data(t_d, load_d['y_ref']); l_load_y_act.set_data(t_d, load_d['y_act'])
  ax_load.set_xlim(max(0, time_elapsed - 5), time_elapsed + 0.1)

  # Visual Simulation (Drones as lines)
  dx, dy = [], []
  visual_arm = 0.1
  for i in range(1, 2 + 1):
    phic = 0.0
    xc, yc = positions_ref[i]
    x1, y1 = xc - visual_arm * np.cos(phic), yc - visual_arm * np.sin(phic)
    x2, y2 = xc + visual_arm * np.cos(phic), yc + visual_arm * np.sin(phic)
    dx.extend([x1, x2, None]); dy.extend([y1, y2, None])

  line_drones.set_data(dx, dy)

  segments = []
  for (_, c) in edges:
    x, y = pulley.x[:,0]
    x_d, y_d = positions_ref[c]
    segments.append([(x_d, y_d), (x, y)])

  cable_collection.set_segments(segments)

  T_abs = np.abs(pulley.T)
  t_min = T_abs.min()
  t_max = max(T_abs.max(), 1e-3)

  cable_collection.set_clim(t_min, t_max)
  cable_collection.set_array(T_abs)

  pt_load.set_data(pulley.x)

  return cable_collection, line_drones, pt_load, l_load_x_act, text_telemetry

ani = FuncAnimation(fig, update, frames=None, interval=Ts*1000, blit=False, cache_frame_data=False)
plt.show()
