import math
import numpy as np

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
                 ):

        # Unit conversion: rpm -> rad/s (gains for ω^2)
        conv = (30.0 / math.pi) ** 2
        self.kf = kf_in_rpm * conv
        self.km = km_in_rpm * conv

        self.L = L
        self.motor_gain = motor_gain
        self.omega_max = (max_omega_rpm * 2.0 * math.pi) / 60.0  # [rad/s]

        # States
        self.omega = np.zeros(2, dtype=float)  # [rad/s]
        self.omega_h = 0.0  # hover baseline [rad/s], set after mass is known

        # 2x2 mixer for desired motor speeds from (ω_h + Dω_F, Dω_θ)
        # [ω1_des, ω2_des]^T = B @ [ω_h + Dω_F, Dω_θ]^T
        self.B = np.array([[1.0, -1.0],
                           [1.0,  1.0]], dtype=float)

    def set_hover_from_mass(self, mass: float):
        """Compute hover speed from mass: mg = 2*kf*ω_h^2  ->  ω_h = sqrt(mg/(2*kf))."""
        self.omega_h = math.sqrt(max(mass * G / (2.0 * self.kf), 0.0))
        return self.omega_h

    # --- mixing helpers ---
    def _mix_from_Dw(self, Domega_F: float, Domega_theta: float):
        v = np.array([self.omega_h + Domega_F, Domega_theta], dtype=float)
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
        self.omega = np.clip(self.omega, 0.0, self.omega_max)

        # Individual thrusts / drag torques
        T_i = self.kf * (self.omega ** 2)   # [N]
        M_i = self.km * (self.omega ** 2)   # [N*m]
        return self.omega.copy(), T_i, M_i

# ------------------------ System Dynamics ------------------------
class SystemDynamic:
  """
  Multibody system dynamics solver for interconnected 2D robots and loads.

  Uses Lagrange multipliers to solve for constrained accelerations.
  """
  def __init__(self,
               num_robots:int,
               agent_masses:list[float] = [10, 5, 5],
               robot_inertia:list[float] = [5e-3, 5e-3],
               Ts:float = 0.01,
               ):
    """
    Initializes system parameters, validates inputs, and pre-allocates matrices.

    Args:
        num_robots (int): Number of robots in the system (must be >= 1).
        agent_masses (list[float]): Masses of each agent in kg. First mass is the load.
        robot_inertia (list[float]): Inertia around the z-axis for each robot in kg*m^2.
        Ts (float): Sample time for the discrete simulation in seconds.
    """
    self.N = num_robots
    self.num_agents = num_robots + 1
    self.Ts = Ts

    # Param validation
    if self.num_agents < 2:
        raise ValueError("<num_robots> must be at least 1.")
    if len(agent_masses) != self.num_agents:
      raise IndexError(f"Length of <agent_masses> must match the number of agents ({self.num_agents}).")
    if len(robot_inertia) != num_robots:
      raise IndexError(f"Length of <robot_inertia> must match the number of robots ({self.N}).")

    self.DOF = 3 * self.N + 2

    # State vectors
    self.q = np.zeros((self.DOF, 1))
    self.d_q = np.zeros((self.DOF, 1))
    self.dd_q = np.zeros((self.DOF, 1))

    # Matrices
    self.A = np.zeros((self.DOF, self.DOF))
    self.Q = np.zeros((self.DOF, 1))
    self.G = np.zeros((self.DOF, 1))
    self.λ = np.zeros((self.N, 1))
    self.J = np.zeros((self.N, self.DOF))
    self.d_J = np.zeros((self.N, self.DOF))
    self.gama = np.zeros((self.N, 1))

    # Pre-allocate the large block matrix for the linear solver
    self.aux_A = np.zeros((self.DOF + self.N, self.DOF + self.N))
    self.aux_B = np.zeros((self.DOF + self.N, 1))

    # Fill constant Mass matrix (A) and Gravity vector (G)
    self.A[0, 0] = agent_masses[0]
    self.A[1, 1] = agent_masses[0]
    self.G[1, 0] = agent_masses[0]*G

    for i, (m, I) in enumerate(zip(agent_masses[1:], robot_inertia)):
      self.A[3*i + 2, 3*i + 2] = m
      self.A[3*i + 3, 3*i + 3] = m
      self.A[3*i + 4, 3*i + 4] = I
      self.G[3*i + 3, 0] = m*G

    # Insert static matrix A into the block matrix aux_A
    self.aux_A[:self.DOF, :self.DOF] = self.A

    # Generate the graph connections (parent, child)
    self.edges = []
    for i in range(1, (self.N // 2) + 1):
      if 2 * i <= self.N + 1:
        self.edges.append((i - 1, 2 * i - 1))
      if 2 * i + 1 <= self.N + 1:
        self.edges.append((i - 1, 2 * i))
    if self.edges == []:
      self.edges.append((0, 1))

  def get_agent_indices(self, idx):
    """Returns the starting index in vector q and its dimension for a given agent."""
    if idx == 0:
        return 0, 2
    else:
      return 2 + (idx - 1) * 3, 3

  def get_states(self):
    """Extracts states into a (3 x N) array for easier plotting/logging."""
    states = np.zeros((3, self.N))
    for i in range(self.N):
      idx_ini, idx_fin = self.get_agent_indices(i)
      agent_state = self.q[idx_ini: idx_fin]
      states[0:len(agent_state), i] = agent_state.T
    return states

  def update_states(self):
    """Integrates accelerations to velocities, and velocities to positions (Euler)."""
    self.d_q += self.dd_q * self.Ts
    self.q += self.d_q * self.Ts
    return self.get_states()

  def update_jacobian(self):
    """Updates the constraint Jacobian (J) and its derivative (d_J)."""
    self.J.fill(0)
    self.d_J.fill(0)

    for k in range(self.N):
      idx_p, idx_c = self.edges[k]
      start_p, _ = self.get_agent_indices(idx_p)
      start_c, _ = self.get_agent_indices(idx_c)

      pos_p = self.q[start_p : start_p + 2]
      pos_c = self.q[start_c : start_c + 2]
      diff = pos_c - pos_p

      self.J[k, start_c : start_c + 2] = 2 * diff.T
      self.J[k, start_p : start_p + 2] = -2 * diff.T

      vel_p = self.d_q[start_p : start_p + 2]
      vel_c = self.d_q[start_c : start_c + 2]
      diff_v = vel_c - vel_p

      self.d_J[k, start_c : start_c + 2] = 2 * diff_v.T
      self.d_J[k, start_p : start_p + 2] = -2 * diff_v.T

    self.gama = self.d_J @ self.d_q

  def apply_collision_avoidance(self, safe_dist=0.5, k_rep=100.0):
    """
    Applies Artificial Potential Field (APF) repulsive forces to the generalized
    force vector (Q) to prevent agents from colliding.
    """
    # Compare every agent against every other agent
    for i in range(self.num_agents):
      for j in range(i + 1, self.num_agents):
        start_i, _ = self.get_agent_indices(i)
        start_j, _ = self.get_agent_indices(j)

        pos_i = self.q[start_i : start_i + 2]
        pos_j = self.q[start_j : start_j + 2]

        diff = pos_i - pos_j
        dist = np.linalg.norm(diff)

        if 0 < dist < safe_dist:
          # Calculate repulsive force magnitude
          force_mag = k_rep * (1.0/dist - 1.0/safe_dist) * (1.0/(dist**2))

          # Direction normal vector
          n = diff / dist

          # Apply forces acting on i (pushing away from j)
          self.Q[start_i : start_i + 2] += force_mag * n
          # Apply equal and opposite forces acting on j
          self.Q[start_j : start_j + 2] -= force_mag * n

  def step(self, F:list[float], τ:list[float], W_load:float= 0.0, safe_dist=0.6):
    """
    Updates the system dynamics for one iteration.

    Args:
        F (list[float]): Thrust force vector, one value for each robot.
        τ (list[float]): Torque vector, one value for each robot.
        W_load (float): x-axis force perturbation for the load.

    Returns:
        tuple: (Current generalized states, Lagrange multipliers)
    """

    # Param validation
    if len(F) != self.N or len(τ) != self.N:
      raise IndexError("Length of F and τ must match num_robots.")

    self.Q.fill(0)
    self.Q[0, 0] = W_load

    for i in range(self.N):
      Fi = F[i]
      τi = τ[i]

      start_idx, _ = self.get_agent_indices(i + 1)
      φi = self.q[start_idx + 2, 0]

      self.Q[start_idx] = - Fi * math.sin(φi)
      self.Q[start_idx + 1] = Fi * math.cos(φi)
      self.Q[start_idx + 2] = τi

    self.apply_collision_avoidance(safe_dist=safe_dist, k_rep=300.0)

    self.update_jacobian()

    self.aux_A[:self.DOF, self.DOF:] = -self.J.T
    self.aux_A[self.DOF:, :self.DOF] = self.J

    self.aux_B[:self.DOF] = self.Q - self.G
    self.aux_B[self.DOF:] = -self.gama

    q_ext = np.linalg.solve(self.aux_A, self.aux_B)

    self.dd_q = q_ext[0:self.DOF]
    self.λ = q_ext[self.DOF:]

    # print(f"shape A:{self.A.shape}")
    # print(f"shape J:{self.J.shape}")
    # print(self.J)
    # print(f"shape gama:{self.gama.shape}")
    # print(f"shape aux_zero:{aux_zero.shape}")
    # print(f"shape aux0:{aux0.shape}")
    # print(f"shape aux1:{aux1.shape}")
    # print(f"shape aux_A:{aux_A.shape}")
    # print(f"shape aux_B:{aux_B.shape}")
    # print(f"shape q_ext:{q_ext.shape}")
    # print(f"shape dd_q:{self.dd_q.shape}")
    # print(f"shape λ:{self.λ.shape}")
    # print(self.λ[:])

    return self.update_states(), self.λ

if __name__ == "__main__":
  sys = SystemDynamic(2)
  sys.q[2:4] = [[1], [1]]   # Drone 1 at (1, 1)
  sys.q[5:7] = [[-1], [1]]  # Drone 2 at (-1, 1)

  states, λ = sys.step([10, 10], [0, 0])
  print("States:\n", states)
