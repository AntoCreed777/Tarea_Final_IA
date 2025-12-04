from collections import deque
from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


C_C = [3,3]
C_T = [0,5]
T_C = [5,0]
T_T = [1,1]


class TidemanChieruzzi(base_strategies):
    """
    Estrategia que, por cada ronda de traiciones del oponente, aumenta en uno
    el número de represalias a aplicar en rondas posteriores.

    Se concede una “nueva oportunidad” al oponente (dos cooperaciones seguidas
    asumiendo que el juego acaba de empezar) si:

    - El oponente está 10 puntos por detrás de esta estrategia.
    - No acaba de comenzar una tanda de traiciones (la última respuesta del oponente fue cooperación).
    - Han pasado al menos 20 rondas desde la última “nueva oportunidad”.
    - Quedan al menos 10 rondas por jugar en el duelo.
    - El total de traiciones registradas se desvía del 50-50 por más de 3 desviaciones estándar
      (criterio basado en la distribución binomial).

    Está adaptada para asumir que la longitud del duelo es 5000 al principio
    del torneo, y luego actualiza la predicción usando el promedio de la longitud
    de los últimos 100 duelos. Esto debido a que no se conoce la longitud final
    del duelo por la implementación del torneo.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.

    Atributos:
        historial_largo_duelo (deque[int]): Historial de duraciones de duelos (máx. 100) para estimar el largo.
        largo_duelo (float): Estimación actual de la longitud del duelo.
        ultima_respuesta_oponente (Elecciones): Última acción tomada por el oponente.
        ultima_respuesta_propia (Elecciones): Última acción tomada por esta estrategia.
        ultima_nueva_oportunidad (int): Última ronda en la que se concedió una “nueva oportunidad”.
        cantidad_de_traiciones_recordadas (int): Traiciones acumuladas desde la última “nueva oportunidad”.
        cantidad_de_cooperaciones (int): Total de cooperaciones del oponente en el duelo actual.
        cantidad_de_traiciones (int): Total de traiciones del oponente en el duelo actual.
        contador_represalias (int): Cantidad de rondas de represalia pendientes.
        puntaje_oponente (int): Acumulador de recompensas del oponente.
        ronda (int): Número de la ronda actual dentro del duelo.
    """

    def __init__(self):
        """
        Inicializa la estrategia estableciendo la última acción observada
        del oponente como cooperación. Llama al inicializador de la clase base.
        """
        super().__init__()

        # Historial para estimar la duración de los duelos (promedio de los últimos 100)
        self.historial_largo_duelo = deque(maxlen=100)
        self.largo_duelo = 5000  # estimación inicial

        self.ultima_respuesta_oponente = Elecciones.COOPERAR
        self.ultima_respuesta_propia = Elecciones.COOPERAR

        self.ultima_nueva_oportunidad = 0  # última ronda en la que se concedió una nueva oportunidad
        self.cantidad_de_traiciones_recordadas = 0
        self.cantidad_de_cooperaciones = 0
        self.cantidad_de_traiciones = 0
        self.contador_represalias = 0
        self.puntaje_oponente = 0
        self.ronda = 0  # ronda actual

    def realizar_eleccion(self) -> Elecciones:
        """
        Evalúa si se cumplen las condiciones para conceder una nueva oportunidad
        al oponente. De no ser el caso, traiciona si hay represalias pendientes
        o coopera si no las hay.
    
        Returns:
            Elecciones: Acción a realizar en la ronda actual.
        """

        # Chequear condiciones para “nueva oportunidad”
        if self._validar_nueva_oportunidad():
    
            # Resetear contadores para darle una nueva oportunidad al oponente
            self.cantidad_de_traiciones_recordadas = 0
            self.contador_represalias = 0

            self.ultima_nueva_oportunidad = self.ronda
            self.ultima_respuesta_propia = Elecciones.COOPERAR

            return Elecciones.COOPERAR  # Primera cooperación

        # Si no se concede una nueva oportunidad
        else:
            if self.contador_represalias > 0:
                # Aplicar represalia pendiente
                self.contador_represalias -= 1
                self.ultima_respuesta_propia = Elecciones.TRAICIONAR
                return Elecciones.TRAICIONAR
            
            else:
                # No hay represalias: cooperar
                self.ultima_respuesta_propia = Elecciones.COOPERAR
                return Elecciones.COOPERAR

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones):
        """
        Registra la cantidad de cooperaciones y traiciones del oponente y
        actualiza su puntaje a partir de la última acción de ambos.

        Args:
            eleccion (Elecciones): Acción tomada por el oponente.
        """

        self.ultima_respuesta_oponente = eleccion

        # Actualizar contadores de cooperaciones y traiciones
        if eleccion == Elecciones.TRAICIONAR:
            self.cantidad_de_traiciones += 1
            self.cantidad_de_traiciones_recordadas += 1

            # Iniciar el contador de represalias al ser traicionado (solo si no había uno activo)
            if self.contador_represalias == 0:
                self.contador_represalias = self.cantidad_de_traiciones_recordadas
        else:
            self.cantidad_de_cooperaciones += 1

        # Calcular puntaje oponente
        if (self.ultima_respuesta_propia == Elecciones.COOPERAR 
            and self.ultima_respuesta_oponente == Elecciones.COOPERAR):
            self.puntaje_oponente += C_C[1]

        elif (self.ultima_respuesta_propia == Elecciones.COOPERAR 
              and self.ultima_respuesta_oponente == Elecciones.TRAICIONAR):
            self.puntaje_oponente += C_T[1]

        elif (self.ultima_respuesta_propia == Elecciones.TRAICIONAR 
              and self.ultima_respuesta_oponente == Elecciones.COOPERAR):
            self.puntaje_oponente += T_C[1]

        else:
            self.puntaje_oponente += T_T[1]

    def notificar_nuevo_oponente(self) -> None:
        """
        Restablece el estado interno al iniciar un enfrentamiento contra
        un oponente nuevo. La estrategia vuelve a asumir cooperación inicial.

        Luego, actualiza la estimación de la longitud del duelo al promedio
        de los largos de los duelos anteriores registrados.
        """

        # Actualizar la estimación de la longitud del duelo al promedio de los anteriores
        self.historial_largo_duelo.append(self.ronda)
        self.largo_duelo = sum(self.historial_largo_duelo) / len(self.historial_largo_duelo)

        # Resetear variables para el nuevo duelo
        self.ultima_respuesta_oponente = Elecciones.COOPERAR
        self.ultima_respuesta_propia = Elecciones.COOPERAR
        self.ultima_nueva_oportunidad = 0

        self.cantidad_de_traiciones_recordadas = 0
        self.cantidad_de_cooperaciones = 0
        self.cantidad_de_traiciones = 0
        self.contador_represalias = 0
        self.puntaje_oponente = 0
        self.ronda = 0

    def recibir_recompensa(self, recompensa: int) -> None:
        """
        Suma la recompensa obtenida en la interacción actual al puntaje total
        y aumenta en uno la cantidad de rondas.

        Args:
            recompensa (int): Valor de recompensa asignado por el motor de
                              simulación según el resultado del turno.
        """
        self.puntaje_torneo_actual += recompensa
        self.puntaje += recompensa
        self.ronda += 1

    def _validar_nueva_oportunidad(self):
        """
        Determina si se puede conceder una nueva oportunidad al oponente.
        Condiciones:
        - El oponente está al menos 10 puntos por detrás de esta estrategia.
        - No se está iniciando una nueva tanda de traiciones (última respuesta del oponente fue cooperación).
        - Han pasado al menos 20 rondas desde la última “nueva oportunidad”.
        - Quedan al menos 10 rondas por jugar en el duelo.
        - El total de traiciones registradas se desvía del 50-50 por más de 3
          desviaciones estándar (distribución binomial).
        """
        valida_nueva_oportunidad = True

        # Deben haber pasado al menos 20 rondas desde la última “nueva oportunidad”
        if self.ronda - self.ultima_nueva_oportunidad <= 20:
            valida_nueva_oportunidad = False

        if valida_nueva_oportunidad:
            # El oponente va perdiendo por al menos 10 puntos.
            puntos_validos = self.puntaje - self.puntaje_oponente >= 10
            # Deben quedar al menos 10 rondas estimadas.
            rondas_validas = self.ronda < self.largo_duelo - 10

            # La última acción del oponente debe ser cooperación (no comienza una tanda de traiciones).
            if puntos_validos and rondas_validas and self.ultima_respuesta_oponente == Elecciones.COOPERAR:
                # El 50-50 se evalúa con distribución binomial: N ensayos, p = 1/2.
                N = self.cantidad_de_cooperaciones + self.cantidad_de_traiciones

                # Desviación estándar: sqrt(N * p * (1 - p)) con p = 1/2 -> sqrt(N)/2
                std_deviation = (N ** (1 / 2)) / 2
                lower = N / 2 - 3 * std_deviation
                upper = N / 2 + 3 * std_deviation

                # Considerar “nueva oportunidad” si las traiciones registradas se salen del rango 3σ.
                if (self.cantidad_de_traiciones_recordadas <= lower or
                    self.cantidad_de_traiciones_recordadas >= upper):
                    return True

        return False

