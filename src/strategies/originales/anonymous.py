from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies
import random


class Anonymous(base_strategies):
    """
    Esta estrategia tiene una probabilidad aleatoria P de cooperar en cada elección.
    La probabilidad de cooperación r es inicialmente 30% y es actualizada cada 10
    movimientos. P es ajustada si el oponente es similar a RANDOM, a muy cooperativa 
    o muy traicionera. P tambien es ajustada en el despues de la ronda 130 si la 
    estrategia tiene un puntaje menos que el oponente. Desafortunadamente el 
    proceso complejo de ajuste causa que P caiga a menudo en el rango de 30% a 70%, 
    lo que hace que parezca aleatoria.

    Debido a esto se implementa una versión simplificada que simplemente elige
    una probabilidad r uniformemente en [0.3, 0.7] en cada elección. Como en el 
    torneo original.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.
    """

    def realizar_eleccion(self) -> Elecciones:
        """
        Elige entre Elecciones.COOPERAR y Elecciones.TRAICIONAR con cierta probabilidad.
        La probabilidad de cooperación r se toma uniformemente en [0.3, 0.7] en cada elección.
        """
        r = random.uniform(3, 7) / 10
        return Elecciones.COOPERAR if random.random() < r else Elecciones.TRAICIONAR

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones):
        # No-op: esta estrategia no adapta su comportamiento según el oponente
        pass

    def notificar_nuevo_oponente(self) -> None:
        # No-op: no mantiene estado entre oponentes
        pass

