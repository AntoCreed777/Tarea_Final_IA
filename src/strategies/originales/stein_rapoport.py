from matplotlib.pylab import chisquare
from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class SteinRapoport(base_strategies):
    """
    Estrategia basada en Tit for Tat (TFT) con tandas de 15 rondas y
    las siguientes modificaciones:

    1. Coopera en las primeras 4 rondas.
    2. Traiciona en las últimas 2 rondas de cada tanda.
    3. Cada 15 rondas aplica el test Chi-cuadrado para verificar si el
       oponente juega aleatoriamente. Si es aleatorio, traiciona.

    Se utiliza un nivel de significancia alfa = 0.05, como en el torneo original.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.

    Atributos:
        ultima_respuesta_oponente (Elecciones):
            Última acción observada del oponente. Inicialmente `Elecciones.COOPERAR`
            y se actualiza tras cada ronda.
        ronda (int): Contador de rondas jugadas.
        alfa (float): Nivel de significancia para el test Chi-cuadrado.
        oponente_es_random (bool): Indica si el oponente parece jugar aleatoriamente.
        traiciones_oponente (int): Cantidad de traiciones del oponente observadas.
        cooperaciones_oponente (int): Cantidad de cooperaciones del oponente observadas.
    """

    def __init__(self):
        """
        Inicializa la estrategia:
        - Establece la última acción del oponente como cooperación.
        - Resetea contadores y parámetros internos.
        - Llama al inicializador de la clase base.
        """
        super().__init__()
        self.ultima_respuesta_oponente = Elecciones.COOPERAR
        self.ronda = 0
        self.alfa = 0.05
        self.oponente_es_random = False

        self.traiciones_oponente = 0
        self.cooperaciones_oponente = 0

    def realizar_eleccion(self) -> Elecciones:
        """
        Determina la acción a ejecutar según el historial del oponente,
        el estado de la tanda actual y el test Chi-cuadrado.

        Returns:
            Elecciones: Acción a realizar (cooperar o traicionar).
        """
        # cooperar las primeras 4 rondas
        if self.ronda <= 4: 
            return Elecciones.COOPERAR
        
        # aplicar TFT hasta la ronda 13 (inicio de tanda: 0..12 TFT, 13..14 traición)
        elif self.ronda % 15 < 13:
            return self.ultima_respuesta_oponente
       
        
        # chequear con Chi-Cuadrado cada 15 rondas
        if self.ronda % 15 == 0:
            _, p_value = chisquare([self.cooperaciones_oponente, self.traiciones_oponente])
            self.oponente_es_random = p_value >= self.alfa

        # traicionar las últimas 2 rondas de cada tanda (13 y 14)
        if self.ronda % 15 in (13, 14):
            return Elecciones.TRAICIONAR
        
        # traicionar si oponente es random
        if self.oponente_es_random:
            return Elecciones.TRAICIONAR
        
        # por defecto TFT
        return self.ultima_respuesta_oponente


    def recibir_eleccion_del_oponente(self, eleccion: Elecciones):
        """
        Registra la acción más reciente del oponente.

        Args:
            eleccion (Elecciones): Acción tomada por el oponente.
        """
        # actualizar contadores del oponente
        if eleccion == Elecciones.TRAICIONAR:
            self.traiciones_oponente += 1
        else:
            self.cooperaciones_oponente += 1

        self.ultima_respuesta_oponente = eleccion

    def notificar_nuevo_oponente(self) -> None:
        """
        Restablece el estado interno al iniciar un enfrentamiento contra
        un nuevo oponente. Asume cooperación inicial y reinicia contadores.
        """
        self.ultima_respuesta_oponente = Elecciones.COOPERAR
        self.ronda = 0
        self.oponente_es_random = False

        self.traiciones_oponente = 0
        self.cooperaciones_oponente = 0

    def recibir_recompensa(self, recompensa):
        """
        Suma la recompensa del turno al puntaje total
        y aumenta en uno el contador de rondas.

        Args:
            recompensa (int): Recompensa asignada por el motor de simulación
                              según el resultado del turno.
        """
        self.puntaje += recompensa
        self.ronda += 1