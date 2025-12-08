from collections import deque

from src.elecciones import Elecciones
from src.strategies.RL.Estados.controlador_de_estados import GestorEstado

#Typing
Jugada = tuple[Elecciones, Elecciones]  # (mi_movimiento, movimiento_oponente)
Estado = tuple[Jugada, ...]  # historial de hasta N jugadas recientes

class HistoryState(GestorEstado):

    def __init__(self, tamaño_estado = 5):

        if tamaño_estado <= 0:
            raise ValueError("tamaño_estado debe ser mayor a 0.")

        self.tamaño_estado = tamaño_estado

        self.estado_inicial()

    def estado_inicial(self):
        self.historial: deque[Jugada] = deque(maxlen=self.tamaño_estado)

    def actualizar_estado(self, mi_eleccion: Elecciones, su_eleccion: Elecciones):
        self.historial.append((mi_eleccion, su_eleccion))
        #return tuple(self.historial)

    def estado_actual(self):
        return tuple(self.historial)