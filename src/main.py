from src.controlador_duelos import ControladorDuelos
from src.selectores_de_oponentes import SelectorPvP, SelectorRandom
from src.strategies import (Davis, Downing, Feld, Grofman, Joss, QLearning,
                            Random, SiempreCoopera, SiempreTraiciona,
                            TitForTat)

if __name__ == "__main__":

    cantidad_de_torneos = 1000
    jugadas_base_duelo = 5000
    limite_de_variacion_de_jugadas = 500

    estrategias = [
        SiempreCoopera(),
        SiempreTraiciona(),
        TitForTat(),
        Random(),
        Davis(),
        Downing(),
        Feld(),
        Grofman(),
        Joss(),
        QLearning(
            tamaño_estado=100,
            alpha=0.2,
            gamma=0.8,
            start_epsilon=0.5,
            end_epsilon=0.1,
            rounds_of_decay_epsilon=int(cantidad_de_torneos * jugadas_base_duelo * 0.4),
        ),
    ]

    torneo = ControladorDuelos(
        estrategias,
        cantidad_de_torneos,
        jugadas_base_duelo,
        limite_de_variacion_de_jugadas,
        selector_de_oponentes=SelectorPvP(),
    )

    torneo.iniciar_duelos()
