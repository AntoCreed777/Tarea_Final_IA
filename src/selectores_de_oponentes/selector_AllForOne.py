import random

from src.selectores_de_oponentes.base_selector_de_oponentes import \
    BaseSelectorDeOponentes
from src.strategies import base_strategies


class SelectorAllForOne(BaseSelectorDeOponentes):
    """
    Selector de oponentes que empareja estrategias de manera aleatoria,
    garantizando que cada estrategia sea usada exactamente una vez por ciclo.
    """

    def __init__(self, protagonista : base_strategies):
        """
        Inicializa el selector con estado de ciclo terminado.
        """
        self.protagonista = protagonista
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
        if len(estrategias) < 1:
            raise ValueError("Debe haber mínimo 1 estrategia para elegir.")

        #duelos = []
        #for estrategia in estrategias:
         #   duelos.append((self.protagonista, estrategia))

       # return duelos

        # Inicializar ciclo si terminó
        if self.termino:
            self.por_enfrentar = estrategias.copy()
            if self.protagonista in self.por_enfrentar:
                self.por_enfrentar.remove(self.protagonista)
            self.termino = False

        # Elegir rival aleatorio de los que quedan
        rival = random.choice(self.por_enfrentar)

        # Eliminar al rival
        self.por_enfrentar.remove(rival)

        # Verificar si el ciclo termina
        if len(self.por_enfrentar) < 1:
            self.termino = True

        enfrentamiento = (self.protagonista, rival)
        return enfrentamiento, self.termino
