from .base_class import base_strategies
from .originales import *
from .RL import  SARSA, QLearning, DeepQNetwork, A2C, DuelingDQN, A2C_LSTM
from .siempre_coopera import SiempreCoopera
from .siempre_traiciona import SiempreTraiciona

__all__ = [
    "base_strategies",
    "SiempreCoopera",
    "SiempreTraiciona",
    "TitForTat",
    "Davis",
    "Downing",
    "Feld",
    "Grofman",
    "Joss",
    "Random",
    "Shubik",
    "SteinRapoport",
    "TidemanChieruzzi",
    "Tullock",
    "Feld", 
    "Graaskamp",
    "Grudger",
    "Nydegger",
    "Anonymous",
    "QLearning",
    "SARSA",
    "DeepQNetwork",
    "A2C",
    "DuelingDQN",
    "A2C_LSTM",
]
