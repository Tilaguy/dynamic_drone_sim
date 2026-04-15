import math

G = 9.81

# ----------------------- Attitude Control (2D) -----------------------
class AttitudeControl2D:
    """
    2D attitude PD controller.

    Calculates the required torque to reach a desired pitch angle and maps it
    to the differential motor speed channel (Dω_θ).
    """

    def __init__(self, L: float, kf: float, omega_hover: float,
                 kp_theta: float = 4.0, kd_theta: float = 2.0):
        self.L   = L
        self.kf  = kf           # N/(rad/s)^2
        self.ω_h = omega_hover  # rad/s
        self.kpθ = kp_theta
        self.kdθ = kd_theta
        self._denom_θ = 4.0 * self.L * self.kf * max(self.ω_h, 1e-9)

    def attitude_channels(self, θ_des, θ, ω):
        """
        Computes the differential speed command for attitude tracking.
        """
        e_θ = θ_des - θ
        τ_des = self.kpθ * e_θ - self.kdθ * ω
        Dω_θ = τ_des / self._denom_θ
        return Dω_θ

# ----------------------- Position Control (2D) -----------------------
class PID:
    """
    PID controller with derivative low-pass filter and back-calculation anti-windup.
    """
    def __init__(self, kp: float, ki: float, kd: float,
                 u_min: float = None, u_max: float = None,  # type: ignore
                 tau_d: float = 0.05, kaw: float = 5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max
        self.tau_d = max(1e-4, tau_d)  # D filter time-constant [s]
        self.kaw   = kaw               # anti-windup gain (back-calculation)

        self.i_term = 0.0
        self.d_filt = 0.0   # filtered derivative term
        self.initialized = False

    def reset(self):
        self.i_term = 0.0
        self.d_filt = 0.0
        self.initialized = False

    def step(self, e: float, de: float, dt: float) -> float:
        """
        Computes the PID step.
        Note: Passing 'de' directly (like -velocity) avoids derivative kick.
        """
        if not self.initialized:
            self.d_filt = de
            self.initialized = True

        # Low-pass filter for derivative
        alpha = self.tau_d / (self.tau_d + max(1e-6, dt))
        self.d_filt = alpha * self.d_filt + (1.0 - alpha) * de

        # P and D terms
        p_term = self.kp * e
        d_term = self.kd * self.d_filt

        # Unsaturated control
        u_unsat = p_term + self.i_term + d_term

        # Saturate output
        u = u_unsat
        if self.u_min is not None: u = max(self.u_min, u)
        if self.u_max is not None: u = min(self.u_max, u)

        # Anti-windup: back-calculation integration
        # i_dot = e + (kaw/ki)*(u - u_unsat)
        if self.ki > 0.0:
            # Matches: I = I + (ki * e + kaw * (u - u_unsat)) * dt
            self.i_term += (self.ki * e + self.kaw * (u - u_unsat)) * dt

        return u

class PositionCascade2D:
    """
    Horizontal -> Altitude -> Attitude cascade controller for a 2D drone.

    Maps position errors to desired accelerations, then to desired pitch
    and collective thrust. Now includes Feedforward terms for dynamic tracking.
    """

    def __init__(self,
                 kp_x=3.0, ki_x=0.2, kd_x=1.5,
                 kp_y=8.0, ki_y=4.0, kd_y=1.0,
                 ax_limit=5.0, ay_limit=5.0,
                 L=0.25, kf=6.11e-8, omega_hover=400.0):

        self.pid_x = PID(kp_x, ki_x, kd_x, u_min=-ax_limit, u_max= ax_limit)
        self.pid_y = PID(kp_y, ki_y, kd_y, u_min=-ay_limit, u_max= ay_limit)

        self.L = L
        self.kf = kf                            # N/(rad/s)^2
        self.ω_h = self.set_hover(omega_hover)  # rad/s

    def set_hover(self, omega_hover: float):
        """Updates internal denominators if mass/hover state changes."""
        self.ω_h = omega_hover
        self._denF = 4.0 * self.kf * max(self.ω_h, 1e-9)
        self._denθ = 4.0 * self.L * self.kf * max(self.ω_h, 1e-9)

    def step(self, xd, x, vx, yd, y, vy, m, dt, ax_ff=0.0, ay_ff=0.0):
        """
        Executes the position cascade control step.

        Args:
            xd, yd: Desired positions.
            x, y: Current positions.
            vx, vy: Current velocities.
            m: Mass of the drone.
            dt: Delta time.
            ax_ff, ay_ff: Feedforward accelerations from trajectory planner.

        Returns:
            Tuple containing (a_x_ref, a_y_ref, theta_des, Dω_F).
        """

        # Horizontal PID -> desired horizontal acceleration
        ex   = xd - x
        evx  = - vx
        a_x_ref = self.pid_x.step(ex, evx, dt) + ax_ff

        # Altitude PID
        ey   = yd - y
        evy  = - vy
        a_y_ref = self.pid_y.step(ey, evy, dt) + ay_ff

        # Allocation: Geometric mapping to theta and Thrust
        num_y = G + a_y_ref
        theta_des = math.atan2(-a_x_ref, num_y)     # rad
        T_cmd     = m * math.hypot(a_x_ref, num_y)  # N

        # Convert thrust to collective channel around hover (ΔF = T_cmd - mg)
        ΔF = T_cmd - (m * G)
        Dω_F = ΔF / self._denF

        return a_x_ref, a_y_ref, theta_des, Dω_F

