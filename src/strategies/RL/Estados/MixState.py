from src.elecciones import Elecciones
from src.strategies.RL.Estados import StatState, HistoryState
from src.strategies.RL.Estados.controlador_de_estados import GestorEstado


class HistoryStatState(GestorEstado):

    def __init__(self, n_grupos= 3, short_memory=5, tamaño_estado = 2):

        self.stat_gestor = StatState(grupos=n_grupos, short_memory=short_memory)
        self.history_gestor = HistoryState(tamaño_estado=tamaño_estado)

        self.n_estados = self.stat_gestor.total_estados() * self.history_gestor.total_estados()
        self.estado_inicial()

    def estado_inicial(self):
        self.stat_gestor.estado_inicial()
        self.history_gestor.estado_inicial()

    def actualizar_estado(self, mi_eleccion: Elecciones, su_eleccion):
        self.stat_gestor.actualizar_estado(mi_eleccion, su_eleccion)
        self.history_gestor.actualizar_estado(mi_eleccion, su_eleccion)

    def estado_actual(self):
        return (self.stat_gestor.estado_actual(), self.history_gestor.estado_actual())