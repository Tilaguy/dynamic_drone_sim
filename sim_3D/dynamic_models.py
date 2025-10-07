import math
import numpy as np
import cupy as cp

G = 9.81

class MotorDynamics:
  """
  Motor dynamics + allocation (4 rotors, 'X' config).
  Entrada: (Dw_f, Dw_phi, Dw_theta, Dw_psi)
  Salida:  W_i (vel. motores), T_i (thrusts), M_i (torques)
  """

  def __init__(self,
               kf_in_rpm=6.11e-8,   # N/(r/min)^2
               km_in_rpm=1.5e-9,    # N*m/(r/min)^2
               motor_gain=20.0,     # 1/s
               max_omega_rpm=7800.0,
               use_gpu=True):
    self.xp = cp if use_gpu else np

    self.__motor_gain = motor_gain

    # Conversión de coeficientes
    conv = (30.0 / math.pi) ** 2
    self.kf = kf_in_rpm * conv
    self.km = km_in_rpm * conv

    # Estados
    self.__motor_speeds = self.xp.zeros(4)  # rad/s

    # Límite físico
    self.__max_omega_rad = (max_omega_rpm * 2.0 * math.pi) / 60.0

    # Matriz de mezcla (4x4)
    self.B = self.xp.array([
        [1,  0,  -1,  1],    # Thrust total
        [1,  1,  0, -1],    # Roll
        [1, 0,  1,  1],    # Pitch
        [1, -1,  0, -1]     # Yaw
    ])

  def get_state(self):
    return {
      "motor_speeds": self.__motor_speeds.copy()
    }

  def update(self, dt, Dw):
    """
    Inputs:
      Dw = [Dw_f, Dw_phi, Dw_theta, Dw_psi] (comandos desde AttitudeControl)
    Outputs:
      (W_i, T_i, M_i)
    """
    Dw = self.xp.array(Dw)

    # Resolver incrementos individuales
    Dw_i = self.B @ Dw

    # Modelo dinámico de primer orden para cada motor
    self.__motor_speeds += self.__motor_gain * (Dw_i - self.__motor_speeds) * dt
    self.__motor_speeds = self.xp.clip(self.__motor_speeds, 0.0, self.__max_omega_rad)

    # Fuerzas y momentos individuales
    T_i = self.kf * (self.__motor_speeds ** 2)
    M_i = self.km * (self.__motor_speeds ** 2)

    return self.__motor_speeds.copy(), T_i, M_i

class RigidBodyDynamics:
  def __init__(self, mass, L=0.25,
               init_orientation=[0.0, 0.0, 0.0],
               init_position=[0.0, 0.0, 0.0],
               use_gpu=True):
    self.xp = cp if use_gpu else np

    self.mass = mass
    self.L = L

    # Inercia (ejemplo simple, puedes ajustarlo a tu modelo real)
    Ixx = 5e-3
    Iyy = 5e-3
    Izz = 1e-2
    self.__I = self.xp.diag([Ixx, Iyy, Izz])
    self.__I_inv = self.xp.linalg.inv(self.__I)

    # Estados
    self.__orientation = self.xp.array(init_orientation)  # roll, pitch, yaw [rad]
    self.__ang_velocity = self.xp.zeros(3)  # [p, q, r]
    self.__position = self.xp.array(init_position)      # [x, y, z]
    self.__lin_velocity = self.xp.zeros(3)  # [vx, vy, vz]

    # Para controladores (valores instantáneos de aceleración)
    self.__lin_acc = self.xp.zeros(3)
    self.__ang_acc = self.xp.zeros(3)

  def get_state(self):
    """Devuelve todos los estados y aceleraciones útiles para los controladores."""
    return {
      "position": self.__position.copy(),
      "velocity": self.__lin_velocity.copy(),
      "orientation": self.__orientation.copy(),
      "ang_velocity": self.__ang_velocity.copy(),
      "lin_acc": self.__lin_acc.copy(),
      "ang_acc": self.__ang_acc.copy()
    }

  def __rotation_matrix(self):
    c_phi, c_theta, c_psi = self.xp.cos(self.__orientation)
    s_phi, s_theta, s_psi = self.xp.sin(self.__orientation)

    return self.xp.array([
      [c_theta * c_psi, c_psi * s_theta * s_phi - s_psi * c_phi, c_psi * s_theta * c_phi + s_psi * s_phi],
      [c_theta * s_psi, s_psi * s_theta * s_phi + c_phi * c_psi, s_psi * s_theta * c_phi - c_psi * s_phi],
      [-s_theta,        c_theta * s_phi,                         c_theta * c_phi]
    ])

  def update(self, dt, T_i, M_i):
    """
    Actualiza la dinámica rígida del dron.
    Entradas:
      - T_i: lista/array con las fuerzas individuales de los motores [N]
      - M_i: lista/array con los momentos de arrastre individuales [N*m]
    """
    T_i = self.xp.array(T_i)
    M_i = self.xp.array(M_i)

    # Fuerzas totales
    F_total = F_total = self.xp.stack([
      self.xp.array(0.0),
      self.xp.array(0.0),
      self.xp.sum(T_i)
      ])


    # Momentos totales (configuración quad en X)
    tau_x = self.L * (T_i[1] - T_i[3])
    tau_y = self.L * (T_i[2] - T_i[0])
    tau_z = (M_i[0] - M_i[1] + M_i[2] - M_i[3])
    M_total = self.xp.array([tau_x, tau_y, tau_z])

    # Dinámica lineal
    R = self.__rotation_matrix()
    gravity = self.xp.array([0.0, 0.0, -self.mass * G])
    self.__lin_acc = (gravity + R @ F_total) / self.mass
    self.__lin_velocity += self.__lin_acc * dt
    self.__position += self.__lin_velocity * dt
    self.__position = self.xp.clip(self.__position, a_min=0, a_max=None)

    # Dinámica angular
    omega = self.__ang_velocity
    self.__ang_acc = self.__I_inv @ (M_total - self.xp.cross(omega, self.__I @ omega))
    self.__ang_velocity += self.__ang_acc * dt

    # Actualización de orientación (Euler integration)
    c_phi, c_theta, c_psi = self.xp.cos(self.__orientation)
    s_phi, s_theta, s_psi = self.xp.sin(self.__orientation)
    t_phi, t_theta, t_psi = self.xp.tan(self.__orientation)
    if abs(c_theta) < 1e-6:
      c_theta = math.copysign(1e-6, c_theta)

    row1 = self.xp.stack([self.xp.array(1.0), s_phi * t_theta, c_phi * t_theta])
    row2 = self.xp.stack([self.xp.array(0.0), c_phi, -s_phi])
    row3 = self.xp.stack([self.xp.array(0.0), s_phi / c_theta, c_phi / c_theta])
    W = self.xp.stack([row1, row2, row3])
    euler_dot = W @ self.__ang_velocity
    self.__orientation += euler_dot * dt

    return self.__orientation, self.__ang_velocity

# ----------------- test harness -----------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    # --- Parámetros físicos ---
    mass = 0.5
    kf = 6.11e-8
    km = 1.5e-9
    L = 0.25

    # --- Instancias de bloques ---
    motors = MotorDynamics(kf_in_rpm=kf, km_in_rpm=km)
    drone = RigidBodyDynamics(mass=mass, L=L)

    # --- Cálculo de hover ---
    hover_w = math.sqrt((mass * G) / (4 * motors.kf))*2
    print("Hover ω [rad/s] =", hover_w)
    print("Thrust total esperado =", 4 * motors.kf * hover_w**2, "N")
    print("Peso =", mass * G, "N")

    # Comando de motores (mantener hover)
    Dw_hover = np.array([hover_w, 0.0, 0.0, 0.0])
    # Nota: aquí simplificamos -> pasamos el hover como Δω_f y cero en los demás

    # --- Simulación ---
    dt = 0.01
    T = 50.0
    steps = int(T / dt)

    z_hist = np.zeros(steps)
    roll_hist = np.zeros(steps)
    pitch_hist = np.zeros(steps)
    yaw_hist = np.zeros(steps)

    for i in range(steps):
        # Bloque motores
        W_i, T_i, M_i = motors.update(dt, Dw_hover)

        # Bloque dinámica rígida
        drone.update(dt, T_i, M_i)

        # Estados
        s = drone.get_state()
        z_hist[i] = s["position"][2]
        roll_hist[i] = s["orientation"][0]
        pitch_hist[i] = s["orientation"][1]
        yaw_hist[i] = s["orientation"][2]

    # --- Gráficas ---
    t = np.linspace(0, T, steps)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(t, z_hist, label="z [m]")
    axs[0].set_title("Evolución de altura")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(t, roll_hist, label="roll")
    axs[1].plot(t, pitch_hist, label="pitch")
    axs[1].plot(t, yaw_hist, label="yaw")
    axs[1].set_title("Evolución de orientación")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()
