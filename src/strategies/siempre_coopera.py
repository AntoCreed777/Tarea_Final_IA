from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class SiempreCoopera(base_strategies):
    """
    Implementación de una estrategia determinista que coopera de manera
    constante en cada interacción.

    Esta estrategia representa el comportamiento clásico conocido como
    "Always Cooperate". No utiliza información del oponente ni ajusta
    su conducta en función de recompensas previas. Su finalidad es servir
    como un agente estable y predecible dentro de entornos de dilema del
    prisionero iterado o juegos similares.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.
    """

    def realizar_eleccion(self) -> Elecciones:
        """
        Retorna la acción de cooperación en todas las circunstancias.

        Returns:
            Elecciones: La acción `Elecciones.COOPERAR`.
        """
        return Elecciones.COOPERAR

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Recibe la elección del oponente. Esta estrategia no utiliza dicha
        información, por lo que no realiza ninguna operación.

        Args:
            eleccion: Elección efectuada por el oponente.
        """
        pass

    def notificar_nuevo_oponente(self) -> None:
        """
        Notifica el inicio de un enfrentamiento con un nuevo oponente.

        La estrategia no mantiene estado entre oponentes, por lo que no
        requiere inicialización adicional.
        """
        pass

    def __str__(self):
        return f"\033[32m{super().__str__()}\033[0m"  # Color Verde
