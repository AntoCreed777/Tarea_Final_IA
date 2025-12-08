from abc import abstractmethod, ABC

from src.elecciones import Elecciones


class GestorEstado(ABC):

    @abstractmethod
    def estado_inicial(self):
        pass

    @abstractmethod
    def actualizar_estado(self, mi_eleccion: Elecciones, su_eleccion: Elecciones) :
        """
        Encargado de actualizar el estado segun la información actual
        :param mi_eleccion: La elección que escogiste
        :param su_eleccion: La elección que escogioel oponente
        """
        pass

    @abstractmethod
    def estado_actual(self):
        """
        :return: El estado actual
        """
        pass