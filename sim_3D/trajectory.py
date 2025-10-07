import numpy as np
import cupy as cp

class PolylineTrajectory3D:
    def __init__(self, points_xyz, times=None, desired_speed=1.0, yaw_mode="tangent"):
        """
        points_xyz: array Nx3 (numpy o cupy)
        times: opcional, Nx. Si None, se asignan por longitud a velocidad constante.
        desired_speed: m/s
        yaw_mode: "tangent" (psi mira en dirección de avance) o "fixed0" (psi=0)
        """
        self.xp = np if isinstance(points_xyz, np.ndarray) else cp
        self.P = points_xyz.astype(float)
        self.N = self.P.shape[0]
        assert self.N >= 2, "Se requieren ≥2 puntos"

        # Longitudes por segmento
        dP = self.P[1:] - self.P[:-1]
        seg_len = self.xp.linalg.norm(dP, axis=1)
        self.seg_len = seg_len
        self.seg_dir = self.xp.divide(dP, seg_len[:, None], where=seg_len[:, None] > 1e-9)

        # Tiempos por segmento (si no vienen, usa v deseada)
        if times is None:
            dt_seg = seg_len / max(desired_speed, 1e-6)
            self.t_nodes = self.xp.concatenate([self.xp.array([0.0]), self.xp.cumsum(dt_seg)])
        else:
            self.t_nodes = times.astype(float)
        self.total_time = float(self.t_nodes[-1])
        self.yaw_mode = yaw_mode

    def _segment_closest(self, A, B, p):
        """ Proyección de p sobre segmento AB. Devuelve punto y parámetro s∈[0,1]. """
        v = B - A
        vv = np.dot(v, v)
        if vv < 1e-12:
            return A, 0.0
        s = float(np.dot(p - A, v) / vv)
        s = max(0.0, min(1.0, s))
        return A + s * v, s

    def closest_state(self, r, t_guess):
        """
        Devuelve:
          rT, vT, aT, t_hat, n_hat, b_hat, psi_T
        """
        xp = np  # trabaja en CPU para estabilidad numérica del control
        P = xp.asarray(self.P)
        r = xp.asarray(r, dtype=float)

        # Busca punto más cercano en todos los segmentos (rápido y robusto)
        best_d2, best_pt, best_i, best_s = 1e30, None, None, None
        for i in range(self.N - 1):
            A, B = P[i], P[i+1]
            q, s = self._segment_closest(A, B, r)
            d2 = float(xp.dot(q - r, q - r))
            if d2 < best_d2:
                best_d2, best_pt, best_i, best_s = d2, q, i, s

        # Estados en el punto más cercano
        rT = best_pt
        dir_seg = P[best_i+1] - P[best_i]
        L = float(xp.linalg.norm(dir_seg))
        t_hat = dir_seg / (L + 1e-9)

        # Velocidad deseada constante por segmento (signo según avance)
        # Aproxima t* por tiempo nodal + s*dt_seg
        dt_seg = float(self.t_nodes[best_i+1] - self.t_nodes[best_i])
        vmag = L / (dt_seg + 1e-9)
        vT = vmag * t_hat

        # Aceleración feed-forward ~ 0 en segmentos rectos
        aT = xp.zeros(3)

        # Normal/binormal: si no hay curvatura, elige una normal estable
        # usando "world up" y Gram-Schmidt
        up = xp.array([0.0, 0.0, 1.0])
        n_tmp = up - xp.dot(up, t_hat) * t_hat
        if xp.linalg.norm(n_tmp) < 1e-6:
            up = xp.array([1.0, 0.0, 0.0])  # fallback
            n_tmp = up - xp.dot(up, t_hat) * t_hat
        n_hat = n_tmp / (xp.linalg.norm(n_tmp) + 1e-9)
        b_hat = xp.cross(t_hat, n_hat)
        b_hat = b_hat / (xp.linalg.norm(b_hat) + 1e-9)

        # Yaw deseado
        if self.yaw_mode == "tangent":
            psi_T = float(np.arctan2(t_hat[1], t_hat[0]))  # yaw desde t_hat en x-y
        else:
            psi_T = 0.0

        return rT, vT, aT, t_hat, n_hat, b_hat, psi_T
