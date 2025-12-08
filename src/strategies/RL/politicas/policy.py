from __future__ import annotations
from abc import ABC, abstractmethod

from src.elecciones import Elecciones

class Policy(ABC):

    @abstractmethod
    def eleccion(self, q_table : QTable, estado : Estado) -> Elecciones:
        """

        :param qtable: Tabla para predeterminar la mejor elcción
        :param estado: Estado actual en el que se está
        :return: La elección a escoger
        """
        pass