import random

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Random(base_strategies):
    """
    Implementación de una estrategia de decisión aleatoria.

    Esta estrategia selecciona una acción de forma uniforme al azar entre todas
    las opciones definidas en la enumeración `Elecciones`. No utiliza memoria,
    historial, ni información del oponente. Su comportamiento es completamente
    estocástico.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.
    """

    def realizar_eleccion(self) -> Elecciones:
        """
        Selecciona y retorna una elección aleatoria.

        Returns:
            Elecciones: Una instancia de la enumeración Elecciones,
            seleccionada mediante muestreo uniforme.
        """
        return random.choice(list(Elecciones))

    def recibir_eleccion_del_oponente(self, eleccion) -> None:
        """
        Recibe la elección del oponente.

        Esta estrategia no utiliza ni almacena esta información, por lo que
        el método no implementa ninguna acción.

        Args:
            eleccion: Elección realizada por el oponente.
        """
        pass

    def notificar_nuevo_oponente(self) -> None:
        """
        Notifica el inicio de una nueva interacción contra un oponente distinto.

        Esta estrategia no mantiene estado entre oponentes, por lo que no
        requiere inicialización ni reseteo.
        """
        pass
