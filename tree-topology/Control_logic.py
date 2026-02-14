import math

G = 9.81

# ----------------------- Attitude Control (2D) -----------------------
class AttitudeControl2D:
    """
    2D attitude PD: outputs (Dω_F, Dω_θ) to feed MotorDynamics2D.
    τ_des = Kp*e_θ - Kd*ω
    Dω_θ  = τ_des / (4*L*kf*ω_h)
    Dω_F  = provided by altitude loop (here PD on y)
    """

    def __init__(self, L, kf, omega_hover,
                 kp_theta=4.0, kd_theta=2.0):
        self.L   = float(L)
        self.kf  = float(kf)           # N/(rad/s)^2
        self.ω_h = float(omega_hover)  # rad/s
        self.kpθ = float(kp_theta)
        self.kdθ = float(kd_theta)
        self._denom_θ = 4.0 * self.L * self.kf * max(self.ω_h, 1e-9)

    def attitude_channels(self, θ_des, θ, ω):
        e_θ = float(θ_des - θ)
        τ_des = self.kpθ * e_θ - self.kdθ * float(ω)
        Dω_θ = τ_des / self._denom_θ
        return Dω_θ


class PID:
    """
    PID with derivative-on-measurement, low-pass on D, and anti-windup by back-calculation.
    Call: step(e, de, dt)  with  e = x_d - x,  de = v_d - v  (or any derivative term you use).
    """
    def __init__(self, kp, ki, kd, u_min=None, u_max=None, tau_d=0.05, kaw=5.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.u_min = u_min
        self.u_max = u_max
        self.tau_d = max(1e-4, float(tau_d))  # D filter time-constant [s]
        self.kaw   = float(kaw)               # anti-windup gain (back-calculation)

        self.i = 0.0
        self.d_filt = 0.0   # filtered derivative term
        self.initialized = False

    def reset(self):
        self.i = 0.0
        self.d_filt = 0.0
        self.initialized = False

    def step(self, e, de, dt):
        if not self.initialized:
            self.d_filt = float(de)
            self.initialized = True

        # Low-pass filter for derivative (first-order)
        alpha = self.tau_d / (self.tau_d + max(1e-6, dt))
        self.d_filt = alpha * self.d_filt + (1.0 - alpha) * float(de)

        # Unsaturated control (for awu back-calculation)
        u_unsat = self.kp * float(e) + self.ki * self.i + self.kd * self.d_filt

        # Saturate
        u = u_unsat
        if self.u_min is not None: u = max(self.u_min, u)
        if self.u_max is not None: u = min(self.u_max, u)

        # Anti-windup: back-calculation (drive integrator to match saturation)
        # i_dot = e + (kaw/ki)*(u - u_unsat)
        if self.ki > 0.0:
            self.i += (float(e) + (self.kaw / self.ki) * (u - u_unsat)) * dt
        else:
            # No integral gain → no anti-windup term needed
            self.i += 0.0

        return u


class PositionCascade2D:
    """
    Horizontal->Altitude->Attitude cascade for 2D.
    - Horizontal PID -> a_x_ref
    - Altitude  PID -> a_y_ref
    - Allocation  -> theta_des, T_cmd
    - Channels    -> (Dω_F, Dω_θ) for motors

    Uso:
      a_x_ref, a_y_ref, theta_des, Dω_F = ctrl.step(xd, x, vx, yd, y, vy, m, kf, ω_h)
      Dω_θ = att.attitude_channels(theta_des, theta, omega)
      motores.update(dt, Dw=(Dω_F, Dω_θ))
    """

    def __init__(self,
                 kp_x=3.0, ki_x=0.2, kd_x=1.5,
                 kp_y=8.0, ki_y=4.0, kd_y=1.0,
                 ax_limit=5.0, ay_limit=5.0,   # [m/s^2] command limits
                 L=0.25, kf=6.11e-8, omega_hover=400.0):
        self.pid_x = PID(kp_x, ki_x, kd_x, u_min=-ax_limit, u_max= ax_limit)
        self.pid_y = PID(kp_y, ki_y, kd_y, u_min=-ay_limit, u_max= ay_limit)
        self.L = float(L)
        self.kf = float(kf)                 # N/(rad/s)^2
        self.ω_h = float(omega_hover)       # rad/s
        self._denF = 4.0 * self.kf * max(self.ω_h, 1e-9)          # for Dω_F
        self._denθ = 4.0 * self.L * self.kf * max(self.ω_h, 1e-9) # for Dω_θ (si quisieras mapear torque directo)

    def set_hover(self, omega_hover):
        self.ω_h = float(omega_hover)
        self._denF = 4.0 * self.kf * max(self.ω_h, 1e-9)
        self._denθ = 4.0 * self.L * self.kf * max(self.ω_h, 1e-9)

    def step(self, xd, x, vx, yd, y, vy, m, dt):
        """
        Returns:
          a_x_ref, a_y_ref, theta_des, Dω_F
        """
        # Horizontal PID -> desired horizontal acceleration
        ex   = float(xd - x)
        evx  = float(0.0 - vx)
        a_x_ref = self.pid_x.step(ex, evx, dt)

        # Altitude PID -> desired vertical acceleration
        ey   = float(yd - y)
        evy  = float(0.0 - vy)
        a_y_ref = self.pid_y.step(ey, evy, dt)

        # Allocation (exact, not small-angle)
        num_y = G + a_y_ref
        theta_des = math.atan2(-a_x_ref, num_y)            # rad
        T_cmd     = float(m) * math.hypot(a_x_ref, num_y) # N

        # Convert thrust to collective channel around hover (ΔF = T_cmd - mg)
        ΔF   = T_cmd - float(m) * G
        Dω_F = ΔF / self._denF

        return a_x_ref, a_y_ref, theta_des, Dω_F

