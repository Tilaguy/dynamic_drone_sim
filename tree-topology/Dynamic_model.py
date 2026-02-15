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

        self.L = float(L)
        self.motor_gain = float(motor_gain)
        self.omega_max = (max_omega_rpm * 2.0 * math.pi) / 60.0  # [rad/s]
        print(f"Ω max per motor: {self.omega_max} [rad/s]")

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
  Docstring for SystemDynamic
  Inputs:  F, τ are (N x 1) vectors
  Outputs: q (3N-1 x 1), λ (N-1 x 1)
  """
  def __init__(self,
               num_robots:int,
               agent_masses:list[float] = [10, 5, 5],
               robot_inertia:list[float] = [5e-3, 5e-3],
               Ts:float = 0.01,
               ):
    """
    Docstring for SystemDynamic.__init__

    :param self: Initializacion, validation of system params and creation of the system matrices
    :param num_robots: Number of robots in the system, must be >= 1
    :type num_robots: int
    :param agent_masses: List of masses of each agent in Kg, the first mass correspond to the load
    :type agent_masses: list[float]
    :param robot_inertia: Listo of inertia aroun z-axis for each robot
    :type robot_inertia: list[float]
    :param Ts: Sample time of the discret system implementation
    :type Ts: float
    """
    self.N = num_robots + 1
    self.Ts = Ts

    # Param validation
    if self.N < 2:
      raise ValueError("The <num_robots> must be at least 1, to be able to simulate the system.")
    elif len(agent_masses) != self.N:
      raise IndexError(f"The lenght of <agent_masses> not match with the number of agets to the system (N = {self.N}).")
    elif len(robot_inertia) != num_robots:
      raise IndexError(f"The lenght of <robot_inertia> not match with the number of robots to the system (num_robots = {num_robots}).")

    # System model construction
    self.DOF = 3*self.N - 1
    self.M = self.N - 1

    self.q = np.zeros((self.DOF, 1))
    self.d_q = np.zeros((self.DOF, 1))
    self.dd_q = np.zeros((self.DOF, 1))

    self.A = np.zeros((self.DOF, self.DOF))
    self.Q = np.zeros((self.DOF, 1))
    self.G = np.zeros((self.DOF, 1))
    self.λ = np.zeros((self.M, 1))
    self.J = np.zeros((self.M, self.DOF))
    self.γ = np.zeros((self.M, 1))

    self.A[0, 0] = agent_masses[0]
    self.A[1, 1] = agent_masses[0]
    self.G[1, 0] = agent_masses[0]*G
    for i, (m, I) in enumerate(zip(agent_masses[1:], robot_inertia)):
      self.A[3*i + 2, 3*i + 2] = m
      self.A[3*i + 3, 3*i + 3] = m
      self.A[3*i + 4, 3*i + 4] = I
      self.G[3*i + 3, 0] = m*G

    # Generate the graph connections
    self.edges = [] # (parent, child)
    for i in range(1, (self.N // 2) + 1):
      if 2 * i <= self.N:
        self.edges.append((i - 1, 2 * i - 1))
      if 2 * i + 1 <= self.N:
        self.edges.append((i - 1, 2 * i))

  def get_states(self):
    states = np.zeros((3, self.N))
    for i in range(self.N):
      idx_ini, idx_fin = self.get_agent_indices(i)
      agent_state = self.q[idx_ini: idx_fin]
      states[0:len(agent_state), i] = agent_state.T
    return states

  def update_states(self):
    self.d_q += self.dd_q * self.Ts
    self.q += self.d_q * self.Ts
    return self.get_states()

  def get_agent_indices(self, idx):
    """ Returns the starting index in the vector q and its dimension """
    if idx == 0:
        return 0, 2
    else:
      return 2 + (idx - 1) * 3, 3

  def update_jacobian(self):
    self.J.fill(0)
    d_J = np.zeros_like(self.J)

    for k in range(self.M):
      idx_p, idx_c = self.edges[k]

      start_p, dim_p = self.get_agent_indices(idx_p)
      start_c, dim_c = self.get_agent_indices(idx_c)

      pos_p = self.q[start_p : start_p + 2]
      pos_c = self.q[start_c : start_c + 2]

      diff = pos_c - pos_p
      self.J[k, start_c : start_c + 2] = 2 * diff.T
      self.J[k, start_p : start_p + 2] = -2 * diff.T

      # update γ
      vel_p = self.d_q[start_p : start_p + 2]
      vel_c = self.d_q[start_c : start_c + 2]
      diff_v = vel_c - vel_p
      d_J[k, start_c : start_c + 2] = 2 * diff_v.T
      d_J[k, start_p : start_p + 2] = -2 * diff_v.T

    self.γ = d_J @ self.d_q

  def step(self,
           F:list[float],
           τ:list[float],
           W_load:float= 0.0
           ):
    '''
    Docstring for step

    :param self: Update the dynamic of the sistem in one iteration
    :param F: Thrust force vector, one value for each robot
    :type F: list[float]
    :param τ: Torque vector, one value for each robot
    :type τ: list[float]
    :param W_load: x-axis force pertubation for load
    :type W_load: float
    '''

    # Param validation
    if len(F) != self.N-1:
      raise IndexError(f"The lenght of <F> not match with the number of robots to the system (num_robots = {self.N-1}).")
    elif len(τ) != self.N-1:
      raise IndexError(f"The lenght of <τ> not match with the number of robots to the system (num_robots = {self.N-1}).")

    self.Q.fill(0)
    self.Q[0, 0] = W_load
    for i in range(self.N-1):
      Fi = F[i]
      τi = τ[i]

      start_idx, _ = self.get_agent_indices(i + 1)
      φi = self.q[start_idx + 2, 0]

      self.Q[start_idx] = - Fi * math.sin(φi)
      self.Q[start_idx + 1] = Fi * math.cos(φi)
      self.Q[start_idx + 2] = τi

    self.update_jacobian()

    # Solve the liear system to predict dd_q and λ
    aux_zero = np.zeros((self.M,self.M))
    aux0 = np.hstack([self.A, -self.J.T])
    aux1 = np.hstack([self.J, aux_zero])
    aux_A = np.vstack([aux0, aux1])
    aux_B = np.vstack([self.Q - self.G, -self.γ])

    q_ext = np.linalg.solve(aux_A, aux_B)

    self.dd_q = q_ext[0:self.DOF]
    self.λ = q_ext[self.DOF:]

    # print(f"shape A:{self.A.shape}")
    # print(f"shape J:{self.J.shape}")
    # print(f"shape γ:{self.γ.shape}")
    # print(f"shape aux_zero:{aux_zero.shape}")
    # print(f"shape aux0:{aux0.shape}")
    # print(f"shape aux1:{aux1.shape}")
    # print(f"shape aux_A:{aux_A.shape}")
    # print(f"shape aux_B:{aux_B.shape}")
    # print(f"shape q_ext:{q_ext.shape}")
    # print(f"shape dd_q:{self.dd_q.shape}")
    # print(f"shape λ:{self.λ.shape}")
    return self.update_states(), self.λ

if __name__ == "__main__":
  sys = SystemDynamic(2)#, [10, 5, 4, 3], [5e-3, 5e-3, 5e-3])
  sys.q[2:4] = [[1], [1]] # Drone 1 en (1, 1)
  sys.q[5:7] = [[-1], [1]] # Drone 2 en (-1, 1)

  states, λ = sys.step([10, 10], [0, 0])
  print(states)
