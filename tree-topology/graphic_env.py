import math
import numpy as np

def generate_initial_pos(N:int, cables_list:list[float], load_position:list[float]):
  M = N-1
  # Param validation
  if N < 2:
    raise ValueError("The <N> must be at least 1 and the load, to be able to simulate the system.")
  elif len(cables_list) !=  M:
    raise IndexError(f"The lenght of <cables_list> not match with the number of edges of the system (num_edges = {M}).")

  edges = [] # (parent, child)
  for i in range(1, (N // 2) + 1):
    if 2 * i <= N:
      edges.append((i - 1, 2 * i - 1))
    if 2 * i + 1 <= N:
      edges.append((i - 1, 2 * i))

  agent_pos = np.zeros((N, 2))
  agent_pos[0, :] = load_position
  for (idx_p, idx_c) in edges:
    xp, yp = agent_pos[idx_p, :]

    xc = xp + cables_list[idx_c-1] * math.cos(math.pi / 3) * (-1 if idx_c%2==0 else 1)
    yc = yp + cables_list[idx_c-1] * math.sin(math.pi / 3)
    agent_pos[idx_c,:] = [xc, yc]

  return agent_pos
