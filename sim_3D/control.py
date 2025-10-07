import numpy as np
import cupy as cp

G = 9.81

class AttitudeControl:
  """
  Attitude controller (Michael et al. 2010, eq. (6)).
  Inputs:  desired Euler angles [phi_d, theta_d, psi_d],
            current Euler angles [phi, theta, psi],
            current angular rates [p, q, r].
  Outputs: [Dw_f, Dw_phi, Dw_theta, Dw_psi]
            For now Dw_f is constant (hover thrust).
  """

  def __init__(self,
               kp_phi=4.0, kd_phi=2.0,
               kp_theta=4.0, kd_theta=2.0,
               kp_psi=2.0, kd_psi=1.0,
               use_gpu=True):
    self.xp = cp if use_gpu else np

    self.kp_phi = kp_phi
    self.kd_phi = kd_phi
    self.kp_theta = kp_theta
    self.kd_theta = kd_theta
    self.kp_psi = kp_psi
    self.kd_psi = kd_psi

  def update(self, desired_angles, current_angles, current_rates):
    phi_d, theta_d, psi_d = desired_angles
    phi, theta, psi = current_angles
    p, q, r = current_rates

    # Errors
    e_phi = phi_d - phi
    e_theta = theta_d - theta
    e_psi = psi_d - psi

    # PD control laws (eq. 6)
    Dw_phi   = self.kp_phi * e_phi   - self.kd_phi * p
    Dw_theta = self.kp_theta * e_theta - self.kd_theta * q
    Dw_psi   = self.kp_psi * e_psi   - self.kd_psi * r

    return self.xp.array([ Dw_phi, Dw_theta, Dw_psi])

class PositionControl:
  def __init__(self,
               m=0.5, kF=6.11e-8, xh=5000.0, num_motors=4,
               kp_x=4.0, ki_x=4.0, kd_x=2.0,
               kp_y=4.0, ki_y=4.0, kd_y=2.0,
               kp_z=2.0, ki_z=2.0, kd_z=1.0,
               use_gpu=True):
    self.xp = cp if use_gpu else np
    # PID gains
    self.kp_x, self.ki_x, self.kd_x = kp_x, ki_x, kd_x
    self.kp_y, self.ki_y, self.kd_y = kp_y, ki_y, kd_y
    self.kp_z, self.ki_z, self.kd_z = kp_z, ki_z, kd_z
    # Parameters
    self.m = m
    self.kF = kF
    self.num_motors = num_motors
    self.xh = xh   # hover rotor speed (rad/s)
    # Integrators
    self.int_ex, self.int_ey, self.int_ez = 0.0, 0.0, 0.0
    self.prev_ex, self.prev_ey, self.prev_ez = 0.0, 0.0, 0.0

  def update_hover(self, desired_angles, desired_position, current_angles, current_position, current_velocity, dt):
    # Unpack states
    _, _, psi_d = desired_angles
    x_d, y_d, z_d = desired_position
    # phi, theta, psi = current_angles
    x, y, z = current_position
    vx, vy, vz = current_velocity

    # Position errors
    ex, ey, ez = x_d - x, y_d - y, z_d - z

    # Integrals
    self.int_ex += ex * dt
    self.int_ey += ey * dt
    self.int_ez += ez * dt

    # Desired accelerations (PID)
    ax_des = self.kp_x*ex + self.kd_x*(-vx) + self.ki_x*self.int_ex
    ay_des = self.kp_y*ey + self.kd_y*(-vy) + self.ki_y*self.int_ey
    az_des = self.kp_z*ez + self.kd_z*(-vz) + self.ki_z*self.int_ez

    # Invert eqs (Michael 2010, eq. 9)
    phi_cmd   = (1/G) * (ax_des*self.xp.sin(psi_d) - ay_des*self.xp.cos(psi_d))
    theta_cmd = (1/G) * (ax_des*self.xp.cos(psi_d) + ay_des*self.xp.sin(psi_d))
    d_omegaF  = (self.m / (2.0*self.num_motors*self.kF*self.xh)) * az_des

    # Return commands for attitude controller
    return phi_cmd, theta_cmd, psi_d, d_omegaF

  def update_track(self,
                 traj, t,
                 current_angles,          # [phi, theta, psi]
                 current_position,        # r
                 current_velocity,        # r_dot
                 yaw_ref=None,            # psi_T(t) si quieres forzarlo
                 clamp_tilt_deg=10.0):
    xp = self.xp

    # 1) Punto más cercano y marco Frenet-Serret en trayectoria
    rT, vT, aT, t_hat, n_hat, b_hat, psi_T = traj.closest_state(current_position, t)

    # 2) Errores (e_p sin componente tangencial; e_v completo)
    e_r = rT - current_position
    ep = xp.dot(e_r, n_hat)*n_hat + xp.dot(e_r, b_hat)*b_hat
    ev = vT - current_velocity

    # 3) Aceleración deseada (PD + feed-forward)
    ax_des = self.kp_x*ep[0] + self.kd_x*ev[0] + aT[0]
    ay_des = self.kp_y*ep[1] + self.kd_y*ev[1] + aT[1]
    az_des = self.kp_z*ep[2] + self.kd_z*ev[2] + aT[2]

    # 4) Yaw deseado
    psi_d = yaw_ref if yaw_ref is not None else psi_T

    # 5) Invertir (9a–9c) → φ, θ, Δω_F (Michael 2010)
    spsi, cpsi = xp.sin(psi_d), xp.cos(psi_d)
    phi_cmd   = (1.0/G) * (ax_des*spsi - ay_des*cpsi)
    theta_cmd = (1.0/G) * (ax_des*cpsi + ay_des*spsi)
    d_omegaF  = (self.m / (2.0*self.num_motors**self.kF*self.xh)) * az_des

    # 6) Clamp de seguridad en tilt
    max_tilt = xp.deg2rad(clamp_tilt_deg)
    phi_cmd   = xp.clip(phi_cmd,   -max_tilt, max_tilt)
    theta_cmd = xp.clip(theta_cmd, -max_tilt, max_tilt)

    return float(phi_cmd), float(theta_cmd), float(psi_d), float(d_omegaF)
