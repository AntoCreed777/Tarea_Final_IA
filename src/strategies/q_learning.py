import random

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies

# Typing
Jugada = tuple[Elecciones, Elecciones]  # (mi_movimiento, movimiento_oponente)
Estado = tuple[Jugada, ...]  # historial de hasta N jugadas recientes
Accion = Elecciones
ValorQ = float

QTable = dict[Estado, dict[Accion, ValorQ]]


class QLearning(base_strategies):
    """
    Estrategia basada en Q-Learning para el dilema del prisionero iterado.

    Esta implementación aprende una política óptima aproximada observando
    las interacciones anteriores. El estado está compuesto por las últimas
    N jugadas (mi movimiento, movimiento del oponente). A partir de ese estado,
    el agente elige una acción usando una política epsilon-greedy y actualiza
    la Q-Table con las recompensas obtenidas.
    """

    def __init__(
        self,
        tamaño_estado: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        """
        Inicializa el agente Q-Learning.

        Parámetros:
        - tamaño_estado (int): cantidad de jugadas previas que definen el estado.
        - alpha (float): tasa de aprendizaje [0,1].
        - gamma (float): descuento futuro para Q-Learning [0,1].
        - epsilon (float): probabilidad de exploración en epsilon-greedy [0,1].
        """
        super().__init__()

        if tamaño_estado <= 0:
            raise ValueError("tamaño_estado debe ser mayor a 0.")

        self.tamaño_estado = tamaño_estado
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self._iniciar_variables()

    def _iniciar_variables(self):
        """
        Reinicia todas las variables internas para comenzar contra un nuevo oponente.
        """
        self.q_table: QTable = {}
        self.historial: list[Jugada] = []
        self.ultimo_estado: Estado | None = None
        self.ultima_accion: Accion | None = None

    def _estado_actual(self) -> Estado:
        """
        Devuelve el estado actual como una tupla de las últimas N jugadas.

        Esto funciona incluso si todavía hay menos de N jugadas registradas.
        """
        return tuple(self.historial[-self.tamaño_estado :])

    def _validar_estado_actual_en_q_table(self, estado: Estado):
        """
        Asegura que el estado exista en la Q-table.
        Si no existe, inicializa sus valores Q con 0.0.
        """
        if estado not in self.q_table:
            self.q_table[estado] = {
                Elecciones.COOPERAR: 0.0,
                Elecciones.TRAICIONAR: 0.0,
            }

    def _elegir_accion(self, estado: Estado) -> Accion:
        """
        Selecciona una acción usando epsilon-greedy:

        - Con prob. epsilon → elige una acción aleatoria (exploración).
        - Si no → elige la acción con mayor valor Q (explotación).
        """
        self._validar_estado_actual_en_q_table(estado)

        if random.random() < self.epsilon:
            return random.choice([Elecciones.COOPERAR, Elecciones.TRAICIONAR])

        acciones = self.q_table[estado]
        return max(acciones, key=acciones.get)

    def _recompensa(self, mi_accion: Elecciones, su_accion: Elecciones) -> float:
        """
        Retorna la recompensa inmediata según las reglas clásicas:

        CC = 3, CT = 0, TC = 5, TT = 1.
        """
        if mi_accion == Elecciones.COOPERAR and su_accion == Elecciones.COOPERAR:
            return 3
        if mi_accion == Elecciones.COOPERAR and su_accion == Elecciones.TRAICIONAR:
            return 0
        if mi_accion == Elecciones.TRAICIONAR and su_accion == Elecciones.COOPERAR:
            return 5
        if mi_accion == Elecciones.TRAICIONAR and su_accion == Elecciones.TRAICIONAR:
            return 1
        return 0

    def realizar_eleccion(self) -> Elecciones:
        """
        Elige una acción para la siguiente jugada, guarda el estado y la acción
        para poder aplicar Q-Learning cuando llegue la respuesta del oponente.
        """
        estado = self._estado_actual()
        accion = self._elegir_accion(estado)

        self.ultimo_estado = estado
        self.ultima_accion = accion

        return accion

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Recibe la jugada del oponente, actualiza el historial,
        calcula la recompensa y aplica la actualización de Q-Learning.
        """
        if self.ultimo_estado is None or self.ultima_accion is None:
            raise ValueError("Valores nulos cuando no deberían serlo.")

        # Registrar la jugada observada
        self.historial.append((self.ultima_accion, eleccion))

        recompensa = self._recompensa(self.ultima_accion, eleccion)
        nuevo_estado = self._estado_actual()

        # Asegurar que el nuevo estado esté inicializado
        self._validar_estado_actual_en_q_table(nuevo_estado)

        # Q-learning update
        q_antiguo = self.q_table[self.ultimo_estado][self.ultima_accion]
        max_q_siguiente = max(self.q_table[nuevo_estado].values())

        nuevo_q = q_antiguo + self.alpha * (
            recompensa + self.gamma * max_q_siguiente - q_antiguo
        )

        # Guardar el nuevo valor Q
        self.q_table[self.ultimo_estado][self.ultima_accion] = nuevo_q

    def notificar_nuevo_oponente(self) -> None:
        """
        Reinicia el agente para un nuevo enfrentamiento.
        """
        self._iniciar_variables()
