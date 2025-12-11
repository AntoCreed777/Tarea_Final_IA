from abc import abstractmethod, ABC

from src.elecciones import Elecciones


class GestorEstado(ABC):

    @abstractmethod
    def estado_inicial(self):
        """
        Regresa el gestor a su estado inicial
        """
        pass

    @abstractmethod
    def actualizar_estado(self, mi_eleccion: Elecciones, su_eleccion: Elecciones) :
        """
        Encargado de actualizar el estado segun la información actual

        Args:

        -mi_eleccion: La elección que escogiste
        -su_eleccion: La elección que escogio el oponente
        """
        pass

    @abstractmethod
    def estado_actual(self):
        """
        return: El estado actual
        """
        pass

    def total_estados(self) -> int:
        """
        Return:
        -int : Total de estados que maneja el gestor
        """
        return self.n_estados