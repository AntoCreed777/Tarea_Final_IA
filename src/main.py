import random
import pandas as pd

from src.controlador_duelos import ControladorDuelos
from src.selectores_de_oponentes import SelectorPvP, SelectorRandom
from src.strategies import (Davis, Downing, Feld, Grofman, Joss,
                            Random, SiempreCoopera, SiempreTraiciona,
                            TitForTat, SARSA, QLearning, DeepQNetwork, A2C, DuelingDQN, A2C_LSTM, Anonymous, Feld, Graaskamp, Grudger, Nydegger, Shubik, SteinRapoport, TidemanChieruzzi, Tullock)
from src.strategies.RL.Estados.MixState import HistoryStatState
from src.strategies.RL.politicas import EpsilonGreedy
from src.strategies.RL.Estados import StatState, HistoryState
from src.strategies.RL.rl import ReinforcementLearning

if __name__ == "__main__":
    # random.seed(42) 
    random.seed(200) 

    cantidad_de_torneos = 18
    jugadas_base_duelo = 100
    limite_de_variacion_de_jugadas = 10

    agente = ReinforcementLearning.load("Agentes/QLearning3.pkl")
    agente.freeze()
    agente.puntaje = 0

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
        agente,
        A2C.load("Agentes/A2C1.pt"),
        #A2C_LSTM.load("Agentes/A2C_LSTM4.pt"),
        #DeepQNetwork.load("Agentes/DeepQNetwork1.pt"),
        #DuelingDQN.load("Agentes/DuelingDQN1.pt")
    ]

    estrategias[-1].freeze()
    # estrategias[-2].freeze()
    # estrategias[-3].freeze()
    # estrategias[-4].freeze()
    # estrategias[-5].freeze()


    torneo = ControladorDuelos(
        estrategias,
        cantidad_de_torneos,
        jugadas_base_duelo,
        limite_de_variacion_de_jugadas,
        selector_de_oponentes=SelectorPvP(),
    )

    analisis = torneo.iniciar_duelos(analisis=True, verbose=True)
    df = pd.DataFrame(analisis)

df.to_csv("resultados/resultados.csv", index=False)
