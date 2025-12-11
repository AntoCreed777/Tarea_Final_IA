from src.elecciones import Elecciones
from src.strategies.RL.rl import ReinforcementLearning


class SARSA(ReinforcementLearning):

    def _iniciar_variables(self):
        """
        Reinicia todas las variables internas para comenzar contra un nuevo oponente.
        """
        super()._iniciar_variables()
        self._siguiente_eleccion()

    def _siguiente_eleccion(self) -> Elecciones:
        """
        Elige una acción para la siguiente jugada, guarda el estado y la acción
        para poder aplicar SARSA cuando llegue la respuesta del oponente.
        """
        estado = self._estado_actual()
        accion = self._elegir_accion(estado)

        self.siguiente_estado = estado
        self.siguiente_accion = accion

        return accion

    def realizar_eleccion(self) -> Elecciones:
        """
        Aplica las decisiones que habia tomado previamente
        """
        self.ultimo_estado = self.siguiente_estado
        self.ultima_accion = self.siguiente_accion

        return self.ultima_accion

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Recibe la jugada del oponente, actualiza el historial,
        calcula la recompensa y aplica la actualización de SARSA.
        """
        if self.ultimo_estado is None or self.ultima_accion is None:
            raise ValueError("Valores nulos cuando no deberían serlo.")

        # Registrar la jugada observada
        self._actualizar_estado(eleccion)

        recompensa = self._recompensa(self.ultima_accion, eleccion)
        nuevo_estado = self._estado_actual()

        # Asegurar que el nuevo estado esté inicializado
        self._validar_estado_actual_en_q_table(nuevo_estado)

        # SARSA update
        q_antiguo = self.q_table[self.ultimo_estado][self.ultima_accion]

        # Decide su siguiente acción para utilizar su valor
        self._siguiente_eleccion()
        q_siguiente = self.q_table[self.siguiente_estado][self.siguiente_accion]

        # Nuevo valor según la verdadera decision tomada
        nuevo_q = q_antiguo + self.alpha * (
                recompensa + self.gamma * q_siguiente - q_antiguo
        )

        # Guardar el nuevo valor Q
        self.q_table[self.ultimo_estado][self.ultima_accion] = nuevo_q

        self.accion_pasada = self.ultima_accion
