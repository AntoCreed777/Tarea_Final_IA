import random

from src.selectores_de_oponentes.base_selector_de_oponentes import \
    BaseSelectorDeOponentes
from src.strategies import base_strategies


class SelectorRandom(BaseSelectorDeOponentes):
    """
    Selector de oponentes que empareja estrategias de manera aleatoria,
    garantizando que cada estrategia sea usada exactamente una vez por ciclo.
    """

    def __init__(self):
        """
        Inicializa el selector con estado de ciclo terminado.
        """
        self.por_enfrentar: list[base_strategies] = []
        self.termino = True

    def seleccionar(self, estrategias):
        """
        Selecciona dos estrategias aleatorias sin repetición dentro del ciclo.

        Parámetros
        ----------
        estrategias : list[base_strategies]
            Lista completa de estrategias disponibles.

        Retorna
        -------
        tuple[(base_strategies, base_strategies), bool]
            - El par seleccionado.
            - True si este fue el último par del ciclo, False si aún quedan.
        """
        if len(estrategias) < 2:
            raise ValueError("Deben haber mínimo 2 estrategias para elegir.")

        # Inicializar ciclo si terminó
        if self.termino:
            self.por_enfrentar = estrategias.copy()
            self.termino = False

        # Elegir par aleatorio de los que quedan
        elecciones = tuple(random.sample(self.por_enfrentar, 2))

        # Eliminar ambos de los disponible
        for eleccion in elecciones:
            self.por_enfrentar.remove(eleccion)

        # Verificar si el ciclo termina
        if len(self.por_enfrentar) < 2:
            self.termino = True

        return elecciones, self.termino
