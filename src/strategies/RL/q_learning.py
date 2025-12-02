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

    Este agente aprende una política aproximada utilizando el algoritmo
    Q-Learning. El estado se define como una tupla que contiene las últimas
    N jugadas observadas, donde cada jugada es un par:
        (mi_movimiento, movimiento_oponente).

    La política utilizada es epsilon-greedy:
    - Con probabilidad epsilon se elige una acción aleatoria (exploración).
    - Con probabilidad (1 - epsilon) se elige la acción con mayor valor Q
      (explotación), según los valores almacenados en la Q-Table.

    Ademas, epsilon decrece de manera lineal desde un epsilon inicial hasta
    un epsilon final. Esto implica que, conforme avanzan las iteraciones,
    el agente reduce progresivamente la exploración y aumenta la
    explotación de los valores Q aprendidos, privilegiando decisiones
    basadas en la experiencia acumulada.
    """

    def __init__(
        self,
        tamaño_estado: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.9,
        start_epsilon: float = 0.5,
        end_epsilon: float = 0.1,
        rounds_of_decay_epsilon: int = 100,
    ):
        """
        Inicializa los hiperparámetros del agente Q-Learning.

        Parámetros:
        - tamaño_estado (int):
            Número de jugadas previas incluidas en el estado (N > 0).

        - alpha (float):
            Tasa de aprendizaje ∈ [0,1]. Controla cuánto se incorporan
            nuevas observaciones a los valores Q existentes.

        - gamma (float):
            Factor de descuento ∈ [0,1] para las recompensas futuras.

        - start_epsilon (float):
            Probabilidad inicial de exploración para la política epsilon-greedy ∈ [0,1].

        - end_epsilon (float):
            Valor mínimo permitido para epsilon al finalizar el decaimiento ∈ [0,1].

        - rounds_of_decay_epsilon (int):
            Cantidad de iteraciones durante las cuales epsilon decrece de
            manera lineal desde start_epsilon hasta end_epsilon.

        Nota importante:
        ----------------
        El parámetro rounds_of_decay_epsilon no se reinicia al comenzar un nuevo duelo.
        Por lo tanto, su valor debe considerarse respecto al número total
        estimado de decisiones que tomará el agente a lo largo de todos los torneos y duelos.

        Recomendación:
        --------------
        Para un torneo compuesto por:
            - T torneos,
            - J jugadas por duelo,

        un valor razonable es:
            rounds_of_decay_epsilon ≈ T * J * 0.4

        Esto permite que la fase de exploración dure una fracción suficiente
        del total de interacciones antes de estabilizarse en end_epsilon.
        """
        super().__init__()

        if tamaño_estado <= 0:
            raise ValueError("tamaño_estado debe ser mayor a 0.")

        self.tamaño_estado = tamaño_estado
        self.alpha = alpha
        self.gamma = gamma

        if start_epsilon < end_epsilon:
            raise ValueError("start_epsilon debe de ser mayor o igual a end_epsilon")

        if rounds_of_decay_epsilon < 1:
            raise ValueError("rounds_of_decay_epsilon debe de ser mayor o igual a 1")

        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.tasa_de_decrecimiento_de_epsilon = (
            start_epsilon - end_epsilon
        ) / rounds_of_decay_epsilon

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

        self.epsilon -= self.tasa_de_decrecimiento_de_epsilon
        self.epsilon = max(self.epsilon, self.end_epsilon)

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

    def get_puntaje_acumulado(self) -> str:
        return (
            "\033[33m" f"{super().get_puntaje_acumulado()}" "\033[0m"
        )  # Color Amarillo

    def get_puntaje_de_este_torneo(self) -> str:
        return (
            "\033[33m" f"{super().get_puntaje_de_este_torneo()}" "\033[0m"
        )  # Color Amarillo
