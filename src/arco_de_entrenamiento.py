import random

from src.controlador_duelos import ControladorDuelos
from src.selectores_de_oponentes import SelectorAllForOne
from src.strategies import (Davis, Downing, Feld, Grofman, Joss, QLearning,
                            Random, SiempreCoopera, SiempreTraiciona,
                            TitForTat, SARSA)

if __name__ == "__main__":
    #random.seed(42)

    cantidad_de_torneos = 100
    jugadas_base_duelo = 500
    limite_de_variacion_de_jugadas = 50

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
    ]

    protas = [
        QLearning(
            tamaño_estado=50,
            alpha=0.2,
            gamma= float(1-(1/jugadas_base_duelo)),
            start_epsilon=0.8,
            end_epsilon=0.4,
            rounds_of_decay_epsilon=int(cantidad_de_torneos * jugadas_base_duelo * 0.4),
        ),
        SARSA(
            tamaño_estado=50,
            alpha=0.2,
            gamma=float(1-(1/jugadas_base_duelo)),
            start_epsilon=0.8,
            end_epsilon=0.4,
            rounds_of_decay_epsilon=int(cantidad_de_torneos * jugadas_base_duelo * 0.4),
        ),
    ]

    for prota in protas:
        print(type(prota))
        enemigos = estrategias.copy()
        enemigos.append(prota) #Para visualizar su puntaje, despues se puede mejorar
        torneo = ControladorDuelos(
                enemigos,
                cantidad_de_torneos,
                jugadas_base_duelo,
                limite_de_variacion_de_jugadas,
                selector_de_oponentes=SelectorAllForOne(prota),
        )
        torneo.iniciar_duelos()
        prota.export_QTable(f"{prota.__class__.__name__}")
