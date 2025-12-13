from abc import ABC, abstractmethod

from src.elecciones import Elecciones



class base_strategies(ABC):
    """
    Clase base abstracta que define la interfaz para todas las estrategias
    utilizadas en simulaciones del dilema del prisionero iterado.

    Esta clase actúa como un contrato obligatorio: toda estrategia debe
    decidir una acción, recibir información del oponente y reaccionar
    apropiadamente. También mantiene un puntaje acumulado otorgado por
    el motor de simulación.
    """

    def __init__(self):
        """
        Inicializa la estrategia con un puntaje acumulado de cero.

        El puntaje refleja el desempeño total de la estrategia a lo largo
        de múltiples interacciones con distintos oponentes.
        """
        self.puntaje = 0
        self.puntaje_torneo_actual = 0
  

    @abstractmethod
    def realizar_eleccion(self) -> Elecciones:
        """
        Determina la acción que realizará la estrategia durante el turno actual.

        La implementación específica depende de la política propia de cada
        estrategia (por ejemplo: cooperar siempre, imitar al oponente,
        traicionar aleatoriamente, etc.).

        Returns:
            Elecciones: La elección realizada por la estrategia, ya sea
                        Elecciones.COOPERAR o Elecciones.TRAICIONAR.
        """
        pass

    @abstractmethod
    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Informa a la estrategia sobre la acción elegida por el oponente en el turno.

        Este método permite que estrategias reactivas puedan adaptar su
        comportamiento según lo ocurrido en turnos anteriores.

        Args:
            eleccion (Elecciones): La elección realizada por el oponente.
        """
        pass

    def recibir_recompensa(self, recompensa: int) -> None:
        """
        Suma la recompensa obtenida en la interacción actual al puntaje total.

        Args:
            recompensa (int): Valor de recompensa asignado por el motor de
                              simulación según el resultado del turno.
        """
        self.puntaje += recompensa
        self.puntaje_torneo_actual += recompensa

    @abstractmethod
    def notificar_nuevo_oponente(self) -> None:
        """
        Notifica a la estrategia que comenzará un enfrentamiento con un nuevo oponente.

        Las estrategias que dependen de historial deben reiniciar aquí
        cualquier estado interno relativo al oponente anterior.
        """
        pass

    def notificar_nuevo_torneo(self):
        self.puntaje_torneo_actual = 0

    def get_puntaje_acumulado(self) -> str:
        return self._formateado_puntaje(self.puntaje)

    def get_puntaje_de_este_torneo(self) -> str:
        return self._formateado_puntaje(self.puntaje_torneo_actual)

    def _formateado_puntaje(self, puntaje: int):
        return f"{type(self).__name__} → {puntaje:,}"
