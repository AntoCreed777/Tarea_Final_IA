from collections import deque

from src.elecciones import Elecciones
from src.strategies.RL.Estados.controlador_de_estados import GestorEstado


class StatState(GestorEstado):

    def __init__(self, grupos = 3, short_memory = 5):
        self.n_grupos = grupos
        self.memory = short_memory
        self.estado_inicial()

    def estado_inicial(self):
        self.jugadas = 0
        self.prob_cooperar_al_cooperar = None
        self.prob_cooperar_al_traicionar = None
        self.prob_cooperar_corto_plazo = None

        self.historial: deque[Elecciones] = deque(maxlen=self.memory)

    def actualizar_estado(self, mi_eleccion: Elecciones, su_eleccion: Elecciones):
        if mi_eleccion is None:
            return

        self.jugadas += 1

        olvidar = None
        if len(self.historial) == self.memory:
            olvidar = self.historial[0]

        # Prepara el factor (1 si aumenta la probabilidad, 0 si disminuye)
        factor_prob_al_cooperar = 0.0
        factor_prob_al_traicionar = 0.0

        # A. Ajuste de la probabilidad a largo plazo (Probabilidad condicional)

        # Si la acción pasada fue COOPERAR, actualizamos self.prob_cooperar_al_cooperar
        if mi_eleccion == Elecciones.COOPERAR:
            if su_eleccion == Elecciones.COOPERAR:
                factor_prob_al_cooperar = 1.0  # El jugador cooperó después de que el oponente cooperara

            # Fórmula de media móvil: new_prob = (old_prob * (n-1) + nuevo_valor) / n

            self.prob_cooperar_al_cooperar = (
                (self.prob_cooperar_al_cooperar * (self.jugadas - 1)) + factor_prob_al_cooperar
            ) / self.jugadas if self.prob_cooperar_al_cooperar is not None else factor_prob_al_cooperar

        # Si la acción pasada fue TRAICIONAR, actualizamos self.prob_cooperar_al_traicionar
        elif mi_eleccion == Elecciones.TRAICIONAR:
            if su_eleccion == Elecciones.COOPERAR:
                factor_prob_al_traicionar = 1.0  # El jugador cooperó después de que el oponente traicionara
            # Si eleccion == TRAICIONAR, factor_prob_al_traicionar queda en 0.0

            # Fórmula de media móvil:
            self.prob_cooperar_al_traicionar = (
                (self.prob_cooperar_al_traicionar * (self.jugadas - 1)) + factor_prob_al_traicionar
            ) / self.jugadas if self.prob_cooperar_al_traicionar is not None else factor_prob_al_traicionar

        if not olvidar:
            self.historial.append(su_eleccion)
            self.prob_cooperar_corto_plazo = self.historial.count(Elecciones.COOPERAR) / len(self.historial)
            return

        cambio_corto_plazo = 0.0

        # Si cooperó, revisamos si olvidamos una TRAICION. Si olvidamos TRAICIONAR, la prob. de COOPERAR aumenta
        if su_eleccion == Elecciones.COOPERAR and olvidar == Elecciones.TRAICIONAR:
            cambio_corto_plazo = (1 / self.memory)

        # Si traicionó, revisamos si olvidamos una COOPERACION. Si olvidamos COOPERAR, la prob. de COOPERAR disminuye
        elif su_eleccion == Elecciones.TRAICIONAR and olvidar == Elecciones.COOPERAR:
            cambio_corto_plazo = -(1 / self.memory)

        self.prob_cooperar_corto_plazo += cambio_corto_plazo

        # 4. Actualización final del historial
        self.historial.append(su_eleccion)

    def estado_actual(self):
        s1 = self._escoger_grupo(self.prob_cooperar_al_cooperar)
        s2 = self._escoger_grupo(self.prob_cooperar_al_traicionar)
        s3 = self._escoger_grupo(self.prob_cooperar_corto_plazo)

        return (s1, s2, s3)

    def _escoger_grupo(self, stat : float):
        if not stat:
            return int((self.n_grupos+1)/ 2)
        grupo = 1
        dist = 1 / self.n_grupos

        while stat > grupo * dist:
            grupo += 1

        return grupo
