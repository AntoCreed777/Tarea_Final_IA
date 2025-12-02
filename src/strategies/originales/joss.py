import random

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Joss(base_strategies):
    """
    Estrategia Joss para el torneo del Dilema del Prisionero iterado.

    Esta estrategia es una variación del famoso Tit-for-Tat, pero con un giro:
    aunque el oponente coopere, Joss puede traicionar con una pequeña probabilidad.

    Comportamiento:
    --------------
    - Primera ronda:
        Se asume que el oponente cooperó (valor inicial por defecto).

    - Si el oponente TRAICIONÓ en la ronda anterior:
        → La estrategia responde con TRAICIÓN.

    - Si el oponente COOPERÓ en la ronda anterior:
        → Coopera con probabilidad 0.9 (9/10).
        → Traiciona con probabilidad 0.1 (1/10).

    Justificación:
    --------------
    Este diseño representa la estrategia Joss clásica:
    una versión ligeramente impredecible de Tit-for-Tat,
    donde suele cooperar, pero mantiene un componente aleatorio que introduce
    incertidumbre y evita ser completamente explotable.

    Atributos:
    ----------
    ultima_eleccion_oponente : Elecciones
        Guarda la última acción realizada por el oponente para decidir la próxima jugada.
    """

    def __init__(self):
        """
        Inicializa la estrategia asumiendo que el oponente cooperó en la ronda anterior.
        Esto permite que la primera elección siga la lógica general de Joss.
        """
        super().__init__()
        self.ultima_eleccion_oponente = Elecciones.COOPERAR

    def realizar_eleccion(self) -> Elecciones:
        """
        Decide la acción de esta ronda según la estrategia Joss.

        Returns
        -------
        Elecciones
            La elección que realiza esta estrategia en la ronda actual.
        """
        # Si el oponente traicionó antes → traicionamos
        if self.ultima_eleccion_oponente == Elecciones.TRAICIONAR:
            return Elecciones.TRAICIONAR

        # Si el oponente cooperó → cooperamos con prob. 0.9
        return (
            Elecciones.COOPERAR if random.random() <= 9 / 10 else Elecciones.TRAICIONAR
        )

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Registra la acción realizada por el oponente en la ronda actual.

        Parameters
        ----------
        eleccion : Elecciones
            Acción realizada por el oponente.
        """
        self.ultima_eleccion_oponente = eleccion

    def notificar_nuevo_oponente(self) -> None:
        """
        Reinicia el estado interno para comenzar contra un oponente nuevo.
        """
        self.ultima_eleccion_oponente = Elecciones.COOPERAR
