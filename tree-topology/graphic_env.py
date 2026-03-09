import math
from cvxpy import pos
import numpy as np
from collections import defaultdict

def generate_initial_position(N:int, cables_len:list[float], load_position:list[float], edges:list[tuple[int, int]]):
  print(cables_len, N)
  # Param validation
  if N < 1:
    raise ValueError("The <N> must be at least 1 and the load, to be able to simulate the system.")
  elif len(cables_len) !=  N:
    raise IndexError(f"The lenght of <cables_len> not match with the number of edges of the system (num_edges = {N}).")

  angle_ref = math.pi / 3.0

  agent_pos = np.zeros((N + 1, 2))
  agent_pos[0, :] = load_position
  for (idx_p, idx_c) in edges:
    xp, yp = agent_pos[idx_p, :]

    xc = xp + cables_len[idx_c-1] * math.cos(angle_ref) * (-1 if idx_c%2==0 else 1)
    yc = yp + cables_len[idx_c-1] * math.sin(angle_ref)
    agent_pos[idx_c,:] = [xc, yc]

  return agent_pos

def generate_position_ref(P, s:list[float]):
  """
  P: (N+1,2)
  s: separation per level
  """

  pos_ref = P.copy()
  N = len(P) - 1
  num_levels = int(math.log2(N))
  # Param validation
  if N < 2:
    raise ValueError("The <N> must be at least 1 and the load, to be able to simulate the system.")
  elif (len(s) !=  num_levels) and (N % 2 != 0) :
    raise IndexError(f"The lenght of <s> not match with the number of robot pairs in the system (robot_pairs = {num_levels}).")

  def level(i):
    return int(math.log2(i + 1))

  # Group by level
  levels = defaultdict(list)
  for i in range(N + 1):
    levels[level(i)].append(i)

  # Process each level
  for lvl in range(1, len(levels)):
    nodes = levels[lvl]
    sep = s[lvl - 1]

    # Order nodes according to current x
    nodes_sorted = sorted(nodes, key=lambda i: pos_ref[i, 0])

    # Left → right sweep
    for k in range(1, len(nodes_sorted)):
      i_prev = nodes_sorted[k - 1]
      i_curr = nodes_sorted[k]

      dx = pos_ref[i_curr, 0] - pos_ref[i_prev, 0]

      # if dx < sep:
      #   pos_ref[i_curr, 0] = pos_ref[i_prev, 0] - sep
      if np.abs(dx) < (sep + 0.8):
        # print(f"\r|dx| = {np.abs(dx)}, S = {sep} + 80%", end="", flush=True)
        pos_ref[i_curr, 0] = pos_ref[i_prev, 0] + sep

  return pos_ref
