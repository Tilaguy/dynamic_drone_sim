class Environment2D:
  def __init__(self, ground_level=0.0, elastic=False, k=500.0, c=20.0):
    """
    ground_level : altura del suelo (z=0 normalmente)
    elastic      : True -> suelo con rebote, False -> suelo rígido sin rebote
    k, c         : constantes del modelo resorte-amortiguador (si elastic=True)
    """
    self.ground_level = ground_level
    self.elastic = elastic
    self.k = k
    self.c = c

  def apply(self, drone_state):
    """
    Aplica restricciones/efectos del suelo al estado del dron.
    drone_state: diccionario de RigidBodyDynamics.get_state()
    """
    pos = drone_state["position"]
    vel = drone_state["velocity"]

    # Chequear colisión con el suelo
    if pos[2] <= self.ground_level:
      if not self.elastic:
        # Caso rígido (clip duro)
        pos[2] = self.ground_level
        if vel[2] < 0:
          vel[2] = 0.0
      else:
        # Caso elástico (rebote con resorte-amortiguador)
        penetration = self.ground_level - pos[2]
        F_ground = self.k * penetration - self.c * vel[2]
        vel[2] += F_ground / drone_state["mass"]  # aplicar aceleración
        pos[2] = self.ground_level  # no permitir hundimiento

    # Devolver estados modificados
    drone_state["position"] = pos
    drone_state["velocity"] = vel
    return drone_state
