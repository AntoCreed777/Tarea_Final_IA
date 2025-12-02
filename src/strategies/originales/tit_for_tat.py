from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class TitForTat(base_strategies):
    """
    Estrategia reactiva basada en el principio de reciprocidad directa.

    Tit for Tat es una estrategia ampliamente estudiada en el dilema del
    prisionero iterado. Su funcionamiento se caracteriza por:

    - Cooperar en la primera interacción.
    - Responder en cada ronda siguiente con la acción ejecutada por el
      oponente en la ronda anterior.

    De este modo, premia la cooperación y castiga la traición de manera
    proporcional y simétrica. No utiliza memoria más allá de la última acción
    recibida, lo que la hace simple, estable y eficaz bajo múltiples
    condiciones experimentales.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.

    Atributos:
        ultima_respuesta_oponente (Elecciones):
            Almacena la última acción tomada por el oponente. Inicia en
            `Elecciones.COOPERAR` y se actualiza tras cada ronda.
    """

    def __init__(self):
        """
        Inicializa la estrategia estableciendo la última acción observada
        del oponente como cooperación. Llama al inicializador de la clase base.
        """
        super().__init__()
        self.ultima_respuesta_oponente = Elecciones.COOPERAR

    def realizar_eleccion(self) -> Elecciones:
        """
        Retorna la acción a ejecutar según el comportamiento del oponente
        en la ronda anterior.

        Returns:
            Elecciones: La última acción del oponente, replicada.
        """
        return self.ultima_respuesta_oponente

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones):
        """
        Actualiza la memoria de la estrategia registrando la acción más reciente
        del oponente.

        Args:
            eleccion (Elecciones): Acción tomada por el oponente.
        """
        self.ultima_respuesta_oponente = eleccion

    def notificar_nuevo_oponente(self) -> None:
        """
        Restablece el estado interno al iniciar un enfrentamiento contra
        un oponente nuevo. La estrategia vuelve a asumir cooperación inicial.
        """
        self.ultima_respuesta_oponente = Elecciones.COOPERAR
