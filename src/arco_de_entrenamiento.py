import random

from src.controlador_duelos import ControladorDuelos
from src.selectores_de_oponentes import SelectorAllForOne
from src.strategies import (Davis, Downing, Feld, Grofman, Joss, QLearning,
                            Random, SiempreCoopera, SiempreTraiciona,
                            TitForTat, SARSA)
from src.strategies.RL.Estados import StatState, HistoryState
from src.strategies.RL.politicas import EpsilonGreedy

if __name__ == "__main__":
    #random.seed(42)

    cantidad_de_torneos = 100
    jugadas_base_duelo = 100
    limite_de_variacion_de_jugadas = 10

    estrategias = [
        TitForTat(),
    ]

    protas = [
        QLearning(
            EpsilonGreedy(start_epsilon=1.0),
            StatState(),
            alpha=0.2,
            gamma= float(1-(1/jugadas_base_duelo))
        ),
        SARSA(
            EpsilonGreedy(start_epsilon=1.0),
            StatState(),
            alpha=0.2,
            gamma=float(1 - (1 / jugadas_base_duelo))
        )
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
        prota.export_QTable(f"{prota.__class__.__name__}+TFT2")
