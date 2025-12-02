import random

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Feld(base_strategies):
    """
    Estrategia Feld para el dilema del prisionero iterado.

    Esta estrategia comienza siendo totalmente cooperativa y, a medida que
    avanzan las rondas, su probabilidad de cooperar decae linealmente hasta
    alcanzar un valor mínimo (end_prob_coop). Además, si el oponente
    traiciona en la ronda anterior, esta estrategia responde inmediatamente
    con una traición, ignorando la parte probabilística.

    Mecánica general:
    -----------------
    - Ronda 1: coopera siempre.
    - Si el oponente traicionó en la última ronda → traiciono.
    - Si el oponente cooperó → coopero con probabilidad decreciente.

    Parámetros:
        start_prob_coop (float): probabilidad inicial de cooperar (típicamente 1.0).
        end_prob_coop (float): probabilidad mínima de cooperación tras el decaimiento.
        rounds_of_decay (int): número de rondas necesarias para descender desde
                               start_prob_coop hasta end_prob_coop.

    Reglas especiales:
        - start_prob_coop debe ser mayor o igual a end_prob_coop.
          (Si no, se lanza ValueError)
    """

    def __init__(
        self,
        start_prob_coop: float = 1.0,
        end_prob_coop: float = 0.5,
        rounds_of_decay: int = 200,
    ):
        super().__init__()
        self.oponent_history: list[Elecciones] = []

        # Validación de parámetros
        if start_prob_coop < end_prob_coop:
            raise ValueError(
                "start_prob_coop debe de ser mayor o igual a end_prob_coop"
            )

        self.start_prob_coop: float = start_prob_coop
        self.end_prob_coop: float = end_prob_coop
        self.rounds_of_decay: int = rounds_of_decay

    def realizar_eleccion(self) -> Elecciones:
        """
        Decide la acción para la ronda actual según las reglas de Feld.

        Reglas:
            - Si es la primera ronda: coopera.
            - Si el oponente traicionó en la ronda anterior: traiciona.
            - En caso contrario: calcula una probabilidad de cooperación que
              decae linealmente con el número de rondas jugadas.

        Retorna:
            Elecciones.COOPERAR o Elecciones.TRAICIONAR
        """
        # Primera ronda: cooperar sí o sí
        if not self.oponent_history:
            return Elecciones.COOPERAR

        # Si el rival traicionó, yo traiciono
        if self.oponent_history[-1] == Elecciones.TRAICIONAR:
            return Elecciones.TRAICIONAR

        # Probabilidad de cooperar
        p = self._cooperation_probability()

        # Elección probabilística
        return Elecciones.COOPERAR if random.random() <= p else Elecciones.TRAICIONAR

    def _cooperation_probability(self) -> float:
        """
        Calcula la probabilidad decreciente de cooperación según la fórmula:

            p(n) = start_prob_coop + (diff / rounds_of_decay) * n

        donde diff = end_prob_coop - start_prob_coop
        y n es el número de rondas transcurridas.

        La probabilidad nunca baja de end_prob_coop.

        Retorna:
            float: probabilidad actual de cooperar.
        """
        diff = self.end_prob_coop - self.start_prob_coop
        slope = diff / self.rounds_of_decay
        rounds = len(self.oponent_history)

        return max(self.start_prob_coop + slope * rounds, self.end_prob_coop)

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Registra la elección realizada por el oponente en la ronda actual.

        Parámetros:
            eleccion (Elecciones): Acción tomada por el rival.
        """
        self.oponent_history.append(eleccion)

    def notificar_nuevo_oponente(self) -> None:
        """
        Reinicia el historial de elecciones cuando comienza un nuevo enfrentamiento.
        """
        self.oponent_history.clear()
