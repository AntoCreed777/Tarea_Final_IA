import random
from itertools import combinations

from src.selectores_de_oponentes.base_selector_de_oponentes import \
    BaseSelectorDeOponentes


class SelectorPvP(BaseSelectorDeOponentes):
    """
    Selector de oponentes que garantiza que cada par de estrategias se enfrente
    exactamente una vez por ciclo. Una vez que todos los pares fueron usados,
    se reinicia automáticamente.
    """

    def __init__(self):
        """
        Inicializa el selector con una lista vacía de pares disponibles.

        La lista se completará automáticamente la primera vez que se llame a
        `seleccionar()`, o cuando se agoten todos los pares posibles.
        """
        self.pares_disponibles: list[tuple] = []

    def seleccionar(self, estrategias):
        """
        Selecciona un par de estrategias que aún no haya sido utilizado
        en el ciclo actual.

        Si `pares_disponibles` está vacío, se genera una nueva lista con
        todas las combinaciones posibles entre las estrategias entregadas.

        Parámetros
        ----------
        estrategias : list
            Lista de instancias de estrategias disponibles.

        Retorna
        -------
        tuple
            Un par de estrategias (e1, e2).
        bool
            True si este fue el último par disponible del ciclo,
            False en caso contrario.
        """
        if not self.pares_disponibles:
            self._generar_pares(estrategias)

        # Seleccionar par aleatorio válido
        par = random.choice(self.pares_disponibles)

        # Eliminar el par para evitar repeticiones
        self.pares_disponibles.remove(par)

        # Indica si el ciclo terminó
        ciclo_terminado = not self.pares_disponibles

        return par, ciclo_terminado

    def _generar_pares(self, estrategias):
        """
        Genera todas las combinaciones posibles de pares entre las
        estrategias recibidas y las almacena en `pares_disponibles`.

        Parámetros
        ----------
        estrategias : list
            Lista de instancias de estrategias desde la cual se arman los pares.
        """
        self.pares_disponibles = list(combinations(estrategias, 2))
