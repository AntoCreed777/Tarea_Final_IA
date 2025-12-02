import random

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Grofman(base_strategies):
    """
    Estrategia Grofman para el Dilema del Prisionero iterado.

    Comportamiento:
    - En la primera ronda siempre coopera.
    - En rondas siguientes, coopera si su última jugada coincide con la del oponente.
    - Si no coinciden, coopera con una probabilidad de 2/7; de lo contrario traiciona.

    Esta estrategia busca ser mayormente cooperativa, pero introduce una ligera
    aleatoriedad que evita caer en ciclos negativos largos.
    """

    def __init__(self):
        """
        Inicializa los historiales propios y del oponente.
        """
        super().__init__()
        self.history: list[Elecciones] = []  # Historial de mis elecciones
        self.oponent_history: list[Elecciones] = []  # Historial del oponente

    def realizar_eleccion(self) -> Elecciones:
        """
        Determina la elección de la estrategia para la ronda actual según las reglas de Grofman.

        Retorna:
            Elecciones: COOPERAR o TRAICIONAR según el comportamiento definido.
        """
        # Primera ronda o coincidencia de elecciones → cooperar
        if len(self.history) == 0 or self.history[-1] == self.oponent_history[-1]:
            move = Elecciones.COOPERAR
        else:
            # Si no coinciden → cooperar con prob. 2/7
            move = (
                Elecciones.COOPERAR
                if random.random() <= 2 / 7
                else Elecciones.TRAICIONAR
            )

        # Registrar mi elección
        self.history.append(move)
        return move

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Registra la elección realizada por el oponente.

        Parámetro:
            eleccion (Elecciones): Acción del rival en esta ronda.
        """
        self.oponent_history.append(eleccion)

    def notificar_nuevo_oponente(self) -> None:
        """
        Resetea los historiales al enfrentar a un nuevo oponente.
        """
        self.history.clear()
        self.oponent_history.clear()
