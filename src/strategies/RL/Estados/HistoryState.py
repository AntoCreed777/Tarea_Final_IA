from collections import deque

from src.elecciones import Elecciones
from src.strategies.RL.Estados.controlador_de_estados import GestorEstado

#Typing
Jugada = tuple[Elecciones, Elecciones]  # (mi_movimiento, movimiento_oponente)
Estado = tuple[Jugada, ...]  # historial de hasta N jugadas recientes

class HistoryState(GestorEstado):
    """
    Clase de tipo GestorEstado

    Busca controlar el estado mediante el historial reciente de las ultimas
    N jugadas del oponente y propias.
    """
    def __init__(self, tamaño_estado = 5):
        """
        Inicializa el historial de estados.

        Args:
            tamaño_estado: Largo del historial que va recordar
        """
        if tamaño_estado <= 0:
            raise ValueError("tamaño_estado debe ser mayor a 0.")

        self.tamaño_estado = tamaño_estado

        self.n_estados = 4**tamaño_estado
        while tamaño_estado >= 3:
            self.n_estados += (4**(tamaño_estado-1)) * 2
            tamaño_estado -= 1
        self.n_estados += 2

        self.estado_inicial()

    def estado_inicial(self):
        """
        Estado inicial es el historial completamente vacio
        """
        self.historial: deque[Jugada] = deque(maxlen=self.tamaño_estado)

    def actualizar_estado(self, mi_eleccion: Elecciones, su_eleccion: Elecciones):
        """
        Actualiza el estado actual al añadir la nueva
        eleccion del oponente y la previa propia.

        Una vez se llene el historial las acciones mas pasadas se eliminan y los
        estados pueden comenzar a repetirse"
        """
        self.historial.append((mi_eleccion, su_eleccion))

    def estado_actual(self):
        """
        Retorna el estado actual en el que esta el agente como una tupla
        de los registros guardados

        Returns:
            El estado en forma de tupla de largo 1 a N
        """
        return tuple(self.historial)
