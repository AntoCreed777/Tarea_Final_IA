import random

from src.controlador_duelos import ControladorDuelos
from src.selectores_de_oponentes import SelectorPvP, SelectorRandom
from src.strategies import (Davis, Downing, Feld, Grofman, Joss,
                            Random, SiempreCoopera, SiempreTraiciona,
                            TitForTat, SARSA, QLearning, DeepQNetwork, A2C, DuelingDQN, A2C_LSTM, Anonymous, Feld, Graaskamp, Grudger, Nydegger, Shubik, SteinRapoport, TidemanChieruzzi, Tullock)
from src.strategies.RL.Estados.MixState import HistoryStatState
from src.strategies.RL.politicas import EpsilonGreedy
from src.strategies.RL.Estados import StatState, HistoryState

if __name__ == "__main__":
    random.seed(42)

    cantidad_de_torneos = 17
    jugadas_base_duelo = 100
    limite_de_variacion_de_jugadas = 10

    estrategias = [
        SiempreCoopera(),
        SiempreTraiciona(),
        Anonymous(),
        Davis(),
        Downing(),
        Feld(),
        Graaskamp(),
        Grofman(),
        Grudger(),
        Joss(),
        Nydegger(),
        Random(),
        Shubik(),
        SteinRapoport(),
        TidemanChieruzzi(),
        TitForTat(),
        Tullock(),
        QLearning(EpsilonGreedy(), HistoryStatState()),
    ]

    #estrategias[-1].import_QTable("QTables/SARSA+4")
    #estrategias[-2].import_QTable("QTables/QLearning+4")
    torneo = ControladorDuelos(
        estrategias,
        cantidad_de_torneos,
        jugadas_base_duelo,
        limite_de_variacion_de_jugadas,
        selector_de_oponentes=SelectorPvP(),
    )

    torneo.iniciar_duelos()
