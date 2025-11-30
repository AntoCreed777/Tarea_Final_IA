from enum import Enum, auto


class Elecciones(Enum):
    """
    Enumeración que representa las posibles decisiones dentro del
    dilema del prisionero iterado.

    Valores:
        COOPERAR:
            Indica que la estrategia elige colaborar con su oponente.

        TRAICIONAR:
            Indica que la estrategia elige no cooperar.
    """

    COOPERAR = auto()
    TRAICIONAR = auto()
