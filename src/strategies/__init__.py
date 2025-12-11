from .base_class import base_strategies
from .originales import Davis, Downing, Feld, Grofman, Joss, TitForTat
from .random import Random
from .RL import  SARSA, QLearning, DeepQNetwork, A2C
from .siempre_coopera import SiempreCoopera
from .siempre_traiciona import SiempreTraiciona

__all__ = [
    "base_strategies",
    "SiempreCoopera",
    "SiempreTraiciona",
    "TitForTat",
    "Random",
    "Davis",
    "Downing",
    "Feld",
    "Grofman",
    "Joss",
    "QLearning",
    "SARSA"
    "DeepQNetwork",
    "A2C",
]
