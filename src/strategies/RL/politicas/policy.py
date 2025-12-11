from __future__ import annotations
from abc import ABC, abstractmethod

from src.elecciones import Elecciones

class Policy(ABC):
    """
    Clase base para una politica de decisión, sirve como contrato obligatorio
    para las politicas, las cuales deben implementar una forma de elegir
    una acción para un estado dado.
    """
    @abstractmethod
    def eleccion(self, q_table : QTable, estado : Estado) -> Elecciones:
        """
        Metodo para escoger una acción según el Estado actual
        y la información en la QTable.

        Args:

        -q_table (QTable): Tabla para predeterminar la mejor elcción

        -estado (Estado): Estado actual en el que se está

        return:
        -Elecciones: La elección a escoger
        """
        pass