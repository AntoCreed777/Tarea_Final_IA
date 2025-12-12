import os
import pickle
from abc import ABC, abstractmethod
from typing import Any

from src.elecciones import Elecciones
from src.strategies.RL.Estados.controlador_de_estados import GestorEstado
from src.strategies.RL.politicas import EpsilonGreedy
from src.strategies.RL.politicas.policy import Policy
from src.strategies.base_class import base_strategies



# Typing
#Jugada = tuple[Elecciones, Elecciones]  # (mi_movimiento, movimiento_oponente)
Accion = Elecciones
ValorQ = float
Estado = Any
QTable = dict[Estado, dict[Accion, ValorQ]]


class ReinforcementLearning( base_strategies ,ABC):

    def __init__(
            self,
            policy: Policy,
            gestor_estado : GestorEstado,
            alpha: float = 0.1,
            gamma: float = 0.9,
    ):
        """
        Inicializa los hiperparámetros del agente Q-Learning.

        Parámetros:

        -policy (Policy):
            Politica que va aplicar el agente para seleccionar sus acciones
            segun el estado en que este (Ej: Epsilon-Greedy).

        -gestor_estado (GestorEstado):
            Instancia de un GestorEstado, se encarga de decidir como cambia
            el estado segun las respustas del entorno a las acciones hechas.

        - alpha (float):
            Tasa de aprendizaje ∈ [0,1]. Controla cuánto se incorporan
            nuevas observaciones a los valores Q existentes.

        - gamma (float):
            Factor de descuento ∈ [0,1] para las recompensas futuras.

        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.policy = policy
        self.gs = gestor_estado

        self.q_table: QTable = {}
        self._estados_alcanzados = 0

        self.old_alpha = self.alpha
        self.old_policy = policy

        self._iniciar_variables()

    def _iniciar_variables(self):
        """
        Reinicia todas las variables internas para comenzar contra un nuevo oponente.
        """
        self.ultimo_estado: Estado | None = None
        self.ultima_accion: Accion | None = None
        self.accion_pasada: Accion | None = None
        self.gs.estado_inicial()

    def _estado_actual(self) -> Estado:
        """
        Devuelve el estado actual .
        """
        return self.gs.estado_actual()

    def _actualizar_estado(self, eleccion: Elecciones):
        """
        Modifica el estado actual

        Args:

        -eleccion: La eleccion hecha por el rival que modifica el estado actual
        """
        self.gs.actualizar_estado(self.accion_pasada, eleccion)

    def _validar_estado_actual_en_q_table(self, estado: Estado):
        """
        Asegura que el estado exista en la Q-table.
        Si no existe, inicializa sus valores Q con 0.0.
        """

        if estado not in self.q_table:
            self._estados_alcanzados += 1
            self.q_table[estado] = {
                Elecciones.COOPERAR: 0.0,
                Elecciones.TRAICIONAR: 0.0,
            }

    def _elegir_accion(self, estado: Estado) -> Accion:
        """
        Selecciona una acción usando la politica dada:
        """
        estado_actual = self._estado_actual()
        self._validar_estado_actual_en_q_table(estado_actual)
        accion = self.policy.eleccion(self.q_table, estado_actual)

        return accion

    def _recompensa(self, mi_accion: Elecciones, su_accion: Elecciones) -> float:
        """
        Retorna la recompensa inmediata según las reglas clásicas:

        CC = 3, CT = 0, TC = 5, TT = 1.
        """
        if mi_accion == Elecciones.COOPERAR and su_accion == Elecciones.COOPERAR:
            return -1
        if mi_accion == Elecciones.COOPERAR and su_accion == Elecciones.TRAICIONAR:
            return -10
        if mi_accion == Elecciones.TRAICIONAR and su_accion == Elecciones.COOPERAR:
            return 0
        if mi_accion == Elecciones.TRAICIONAR and su_accion == Elecciones.TRAICIONAR:
            return -6
        return 0

    def notificar_nuevo_oponente(self) -> None:
        """
        Reinicia el agente para un nuevo enfrentamiento.
        """
        self._iniciar_variables()

    def save(self, file : str) -> None:
        """
        Exporta la QTable para futuros agentes
        """
        # Crear carpeta si no existe
        os.makedirs("QTables", exist_ok=True)

        with open(f"Qtables/{file}.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod 
    def load(path):
        with open(path, 'rb') as f: 
            return pickle.load(f)
        
    def porcentaje_explorado(self) -> float:
        """
        Función para calcular la proporción de estados alcanzados en total

        Returns:
            float: proporcion de estados alcanzados en total
        """
        return self._estados_alcanzados / self.gs.total_estados()

    def freeze(self):
        """
        Congela el aprendizaje del agente y guarda su configuración
        de aprendizaje
        """
        #Guardar su configuración por si se requiere descongelar el entrenamiento
        self.old_alpha = self.alpha
        self.old_policy = self.policy

        #Se setean parametros de decision para solo tomar las mejores decisiones y sin aprender.
        self.alpha = 0
        self.policy = EpsilonGreedy(0,0,1)

    def unfreeze(self):
        """

        Descongela el aprendizaje del agente y recupera su configuración,
        en caso de que no se haya congelado previamente no pasará nada.
        """
        #Se recupera la configuración pasada
        self.alpha = self.old_alpha
        self.policy = self.old_policy