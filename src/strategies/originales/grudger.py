from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Grudger(base_strategies):
    """
    Estrategia Grudger:
    - Comienza cooperando.
    - Si el oponente traiciona en cualquier momento, traiciona de ahí en adelante.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.

    Atributos:
        ha_sido_traicionado (bool): Indica si el oponente ha traicionado al menos una vez.
    """

    def __init__(self):
        """
        Inicializa la estrategia:
        - Llama al inicializador de la clase base.
        - Resetea el estado de traición del oponente.
        """
        super().__init__()
        self.ha_sido_traicionado = False

    def realizar_eleccion(self) -> Elecciones:
        """
        Devuelve la acción a realizar:
        - Cooperar si no ha sido traicionado.
        - Traicionar desde el primer momento en que el oponente traicione, hasta el final.

        Returns:
            Elecciones: COOPERAR o TRAICIONAR según el estado interno.
        """
        if self.ha_sido_traicionado: 
            return Elecciones.TRAICIONAR

        return Elecciones.COOPERAR

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones):
        """
        Registra la última acción del oponente y actualiza el estado interno.

        Args:
            eleccion (Elecciones): Acción tomada por el oponente.
        """
        if eleccion == Elecciones.TRAICIONAR:
            self.ha_sido_traicionado = True

    def notificar_nuevo_oponente(self) -> None:
        """
        Restablece el estado interno al iniciar contra un nuevo oponente:
        - Limpia la bandera de traición previa.
        - Asume cooperación inicial.
        """
        self.ha_sido_traicionado = False
