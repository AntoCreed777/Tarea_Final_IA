from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies
from collections import deque
import random


class Tullock(base_strategies):
    """
    Estrategia Tullock:
    - Coopera en las primeras 11 rondas.
    - A partir de entonces, coopera con probabilidad:
      max(0, proporción de cooperación del oponente en sus últimas hasta 10 rondas - 0.10).

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.

    Atributos:
        ultima_respuesta_oponente (Elecciones):
            Última acción tomada por el oponente.
        _historial_oponente (deque):
            Historial de las últimas hasta 10 acciones del oponente.
        rondas_a_cooperar (int):
            Cantidad de rondas iniciales en las que se coopera (11).
        ronda (int):
            Contador de rondas transcurridas contra el oponente actual.
    """

    def __init__(self):
        """
        Inicializa el estado interno:
        - Supone cooperación inicial del oponente.
        - Crea el historial (máx. 10) de acciones del oponente.
        - Configura 11 rondas de cooperación inicial.
        - Resetea el contador de ronda.
        """
        super().__init__()
        self.ultima_respuesta_oponente = Elecciones.COOPERAR
        self._historial_oponente = deque(maxlen=10)
        self.rondas_a_cooperar = 11  # primeras 11 rondas coopera
        self.ronda = 0

    def realizar_eleccion(self) -> Elecciones:
        """
        Devuelve la acción de esta ronda:
        - Si aún está dentro de las primeras 11 rondas, coopera.
        - En caso contrario, calcula la proporción de cooperaciones del oponente
          en sus últimas hasta 10 acciones y coopera con probabilidad
          max(0.0, proporción - 0.10); de lo contrario, traiciona.
        """
        # Cooperar en las primeras 'rondas_a_cooperar' rondas
        if self.ronda < self.rondas_a_cooperar:
            return Elecciones.COOPERAR

        # A partir de la ronda 12, calcular probabilidad basada en historial
        else:
            # Usar el tamaño real del historial (hasta 10) en lugar de dividir por 10 fijo
            total = len(self._historial_oponente)
            if total == 0:
                prob_cooperar = 0.0
            else:
                cooperaciones = sum(1 for e in self._historial_oponente if e == Elecciones.COOPERAR)
                prop_cooperar = cooperaciones / total
                prob_cooperar = max(0.0, prop_cooperar - 0.10)

            # Elegir acción por probabilidad
            return Elecciones.COOPERAR if random.random() < prob_cooperar else Elecciones.TRAICIONAR

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones):
        """
        Registra la acción del oponente:
        - Actualiza la última respuesta.
        - Añade la acción al historial (ventana de 10).
        - Incrementa el contador de ronda.
        """
        self.ultima_respuesta_oponente = eleccion
        # Actualizar historial y avanzar ronda
        self._historial_oponente.append(eleccion)
        self.ronda += 1

    def notificar_nuevo_oponente(self) -> None:
        """
        Reinicia el estado para un nuevo oponente:
        - Supone cooperación inicial.
        - Limpia el historial de acciones del oponente.
        - Reinicia el contador de ronda a 0.
        """
        self.ultima_respuesta_oponente = Elecciones.COOPERAR
        self._historial_oponente.clear()
        self.ronda = 0
