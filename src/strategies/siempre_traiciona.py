from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class SiempreTraiciona(base_strategies):
    """
    Estrategia determinista que ejecuta la acción de traición en cada ronda.

    Esta implementación corresponde al comportamiento clásico conocido como
    "Always Defect". El agente nunca coopera, independientemente del historial,
    recompensas previas o conducta del oponente. Está diseñada para servir como
    una referencia base en simulaciones de dilema del prisionero iterado u
    otros juegos de interacción repetida.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.
    """

    def realizar_eleccion(self) -> Elecciones:
        """
        Retorna siempre la acción de traición.

        Returns:
            Elecciones: La acción `Elecciones.TRAICIONAR`.
        """
        return Elecciones.TRAICIONAR

    def recibir_eleccion_del_oponente(self, eleccion):
        """
        Recibe la elección del oponente, pero esta estrategia no utiliza
        dicha información ni modifica su comportamiento.

        Args:
            eleccion: La elección realizada por el oponente.
        """
        pass

    def notificar_nuevo_oponente(self) -> None:
        """
        Notifica el inicio de una interacción contra un nuevo oponente.

        Dado que esta estrategia no mantiene estado ni memoria, no requiere
        realizar ningún tipo de inicialización adicional.
        """
        pass
