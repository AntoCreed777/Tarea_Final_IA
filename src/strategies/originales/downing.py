from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Downing(base_strategies):
    """
    Implementación de la estrategia Downing para el Dilema del Prisionero Iterado.

    Esta estrategia aprende si el oponente responde de manera “buena” o “mala”
    dependiendo de si coopera después de que este agente coopera o traiciona.
    Con base en estas métricas, ajusta probabilísticamente su comportamiento para
    decidir entre cooperar, traicionar o alternar.

    Atributos:
        round_number (int): Número de ronda actual.
        history (list[Elecciones]): Historial de elecciones propias.
        oponent_history (list[Elecciones]): Historial de elecciones del oponente.

        good (float): Métrica que mide cuán “bueno” es el rival (coopera tras mi cooperación).
        bad (float): Métrica de cuán “malo” es el rival (coopera tras mi traición).

        nice1 (int): Conteo de cooperaciones del rival tras mi cooperación.
        nice2 (int): Conteo de cooperaciones del rival tras mi traición.

        total_COOPERACIONES (int): Total de veces que este agente ha cooperado (para estadística).
        total_TRAICIONES (int): Total de veces que este agente ha traicionado (para estadística).
    """

    def __init__(self):
        """
        Inicializa el estado interno de la estrategia Downing.
        """
        super().__init__()
        self.round_number = 0
        self.history: list[Elecciones] = []
        self.oponent_history: list[Elecciones] = []

        self.good = 1.0
        self.bad = 0.0
        self.nice1 = 0
        self.nice2 = 0
        self.total_COOPERACIONES = 0
        self.total_TRAICIONES = 0

    def realizar_eleccion(self) -> Elecciones:
        """
        Decide la acción para la ronda actual según la estrategia Downing.

        Retorna:
            Elecciones: La acción seleccionada (COOPERAR o TRAICIONAR).
        """
        self.round_number += 1

        # Primer movimiento: siempre cooperar.
        if self.round_number == 1:
            move = Elecciones.COOPERAR
            self.history.append(move)
            return move

        # A partir de la tercera ronda se puede evaluar respuesta del rival.
        if len(self.history) >= 2:
            # Si hace dos rondas traicioné...
            if self.history[-2] == Elecciones.TRAICIONAR:
                # ...y el rival cooperó en la ronda pasada, cuenta como “nice2”
                if self.oponent_history[-1] == Elecciones.COOPERAR:
                    self.nice2 += 1
                self.total_TRAICIONES += 1
                self.bad = self.nice2 / self.total_TRAICIONES
            else:
                # Si cooperé hace dos rondas y el rival también cooperó en la última
                if self.oponent_history[-1] == Elecciones.COOPERAR:
                    self.nice1 += 1
                self.total_COOPERACIONES += 1
                self.good = self.nice1 / self.total_COOPERACIONES

        # Cálculo de métricas de decisión
        c = 6.0 * self.good - 8.0 * self.bad - 2
        alt = 4.0 * self.good - 5.0 * self.bad - 1

        # Selección basada en los valores calculados
        if c >= 0 and c >= alt:
            move = Elecciones.COOPERAR
        elif (c >= 0 and c < alt) or (alt >= 0):
            move = self.history[-1]
        else:
            move = Elecciones.TRAICIONAR

        self.history.append(move)
        return move

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Registra la elección realizada por el oponente.

        Parámetro:
            eleccion (Elecciones): Acción del rival en la ronda actual.
        """
        self.oponent_history.append(eleccion)

    def notificar_nuevo_oponente(self) -> None:
        """
        Resetea todos los contadores y estados internos para enfrentar un nuevo oponente.
        """
        self.round_number = 0
        self.good = 1.0
        self.bad = 0.0
        self.nice1 = 0
        self.nice2 = 0
        self.total_COOPERACIONES = 0
        self.total_TRAICIONES = 0
        self.history.clear()
        self.oponent_history.clear()
