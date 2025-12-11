from src.elecciones import Elecciones
from src.strategies.RL.rl import ReinforcementLearning


class QLearning(ReinforcementLearning):

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
        self._actualizar_estado(eleccion)

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

        self.accion_pasada = self.ultima_accion
