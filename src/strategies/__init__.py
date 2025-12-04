from .base_class import base_strategies
from .originales import *

from .RL import QLearning
from .siempre_coopera import SiempreCoopera
from .siempre_traiciona import SiempreTraiciona

__all__ = [
    "base_strategies",
    "SiempreCoopera",
    "SiempreTraiciona",
    "QLearning",
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
    "Anonymous"
]