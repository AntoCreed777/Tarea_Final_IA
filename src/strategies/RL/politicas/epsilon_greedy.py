import random
from src.strategies.RL.politicas.policy import *


class EpsilonGreedy(Policy):

    def __init__(self,
            start_epsilon: float = 0.5,
            end_epsilon: float = 0.1,
            rounds_of_decay_epsilon: int = 100):
        """
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
        """

        if start_epsilon < end_epsilon:
            raise ValueError("start_epsilon debe de ser mayor o igual a end_epsilon")

        if rounds_of_decay_epsilon < 1:
            raise ValueError("rounds_of_decay_epsilon debe de ser mayor o igual a 1")

        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.tasa_de_decrecimiento_de_epsilon = (
            start_epsilon - end_epsilon
        ) / rounds_of_decay_epsilon

    def eleccion(self, q_table, estado) -> Elecciones:
        """

        :param qtable: Tabla para predeterminar la mejor elcción
        :param estado: Estado actual en el que se está
        :return: La elección a escoger
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
        if random.random() < self.epsilon:
            return random.choice([Elecciones.COOPERAR, Elecciones.TRAICIONAR])

        acciones = q_table[estado]

        self.epsilon -= self.tasa_de_decrecimiento_de_epsilon
        self.epsilon = max(self.epsilon, self.end_epsilon)

        return max(acciones, key=acciones.get)

