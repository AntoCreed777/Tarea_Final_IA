from abc import ABC, abstractmethod

from src.strategies import base_strategies


class BaseSelectorDeOponentes(ABC):
    """
    Clase base abstracta para la definición de políticas de selección de oponentes
    dentro del sistema de enfrentamientos entre estrategias.
    """

    @abstractmethod
    def seleccionar(
        self, estrategias: list[base_strategies]
    ) -> tuple[tuple[base_strategies, base_strategies], bool]:
        """
        Define la interfaz para la selección de dos estrategias que participarán
        en un duelo.

        Esta función debe ser implementada por todas las subclases y representa
        el núcleo de cualquier política de emparejamiento. Su objetivo es
        encapsular la lógica que determina qué dos instancias de estrategias
        serán enfrentadas en una ronda, permitiendo sustituir o extender
        comportamientos sin modificar el controlador de duelos.

        Parámetros
        ----------
        estrategias : list[base_strategies]
            Colección de estrategias disponibles entre las cuales se deben
            seleccionar dos participantes distintos.

        Retorna
        -------
        tuple[base_strategies, base_strategies]
            Par que contiene las dos estrategias seleccionadas.
        bool
            Si ya termino de generar todas las convinaciones posibles para
            el torneo. Si es False se debe de llamar de nuevo

        Notas
        -----
        Las implementaciones concretas deben garantizar que las dos estrategias
        retornadas sean instancias válidas y diferentes entre sí.
        """
        raise NotImplementedError()
