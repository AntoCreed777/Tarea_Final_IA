from .base_class import base_strategies
from .davis import Davis
from .random import Random
from .siempre_coopera import SiempreCoopera
from .siempre_traiciona import SiempreTraiciona
from .tit_for_tat import TitForTat

__all__ = [
    "base_strategies",
    "SiempreCoopera",
    "SiempreTraiciona",
    "TitForTat",
    "Random",
    "Davis",
]
