from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Davis(base_strategies):
    """
    Implementación de la estrategia conocida como 'Davis' o 'Tit for Tat with
    initial cooperation', utilizada en el dilema del prisionero iterado.

    Esta estrategia se caracteriza por un comportamiento inicial altamente
    cooperativo durante un número predefinido de rondas. Tras ese periodo:

    - Si el oponente ha traicionado al menos una vez, la estrategia responde
      con traición permanentemente.
    - Si el oponente nunca ha traicionado, mantiene la cooperación.

    El objetivo es identificar oponentes cooperativos durante las primeras
    rondas y castigar a aquellos que demuestran intenciones hostiles.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.

    Atributos:
        rondas_a_cooperar (int):
            Número de rondas iniciales de cooperación incondicional.
        ronda_actual (int):
            Contador de rondas disputadas en el duelo actual.
        traiciones_del_oponente (int):
            Contador de acciones de traición realizadas por el oponente.
        puntaje (int):
            Acumulador de recompensas heredado de la clase base.
    """

    def __init__(self, rondas_a_cooperar: int = 10):
        """
        Inicializa la estrategia Davis con el número definido de rondas de
        cooperación inicial, además de sus contadores internos.

        Args:
            rondas_a_cooperar (int): Cantidad de rondas iniciales en las que
                                     se coopera sin condiciones.
        """
        super().__init__()
        self.rondas_a_cooperar = rondas_a_cooperar
        self.ronda_actual = 0
        self.traiciones_del_oponente = 0

    def realizar_eleccion(self) -> Elecciones:
        """
        Determina la acción de la estrategia en la ronda actual siguiendo las
        reglas de Davis:

                - Durante las primeras `rondas_a_cooperar` rondas, se coopera incondicionalmente.
        - Una vez superado ese umbral:
            * Si el oponente ha traicionado alguna vez, se traiciona.
            * En caso contrario, se mantiene la cooperación.

        Returns:
            Elecciones: Acción seleccionada para la ronda actual.
        """
        # Avanzar contador de ronda
        self.ronda_actual += 1

        # Cooperación incondicional durante las primeras N rondas
        if self.ronda_actual <= self.rondas_a_cooperar:
            return Elecciones.COOPERAR

        # Tras el periodo inicial: si el oponente traicionó alguna vez, traicionar
        if self.traiciones_del_oponente > 0:
            return Elecciones.TRAICIONAR

        # De lo contrario, seguir cooperando
        return Elecciones.COOPERAR

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Actualiza los contadores internos según la acción ejecutada por el
        oponente en la ronda actual.

        Args:
            eleccion (Elecciones): Acción tomada por el oponente.
        """
        if eleccion == Elecciones.TRAICIONAR:
            self.traiciones_del_oponente += 1

    def notificar_nuevo_oponente(self) -> None:
        """
        Restablece el estado interno para comenzar la interacción contra un
        oponente completamente nuevo, reiniciando los contadores de acciones
        observadas.
        """
        self.ronda_actual = 0
        self.traiciones_del_oponente = 0
