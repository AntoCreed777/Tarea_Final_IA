import random
from collections import deque
from matplotlib.pylab import chisquare
from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Graaskamp(base_strategies):
    """
    Juega Tit for Tat (TFT) durante 50 movimientos, traiciona en el 51, y luego
    juega 5 movimientos más de TFT. Luego:
    - Si el oponente parece RANDOM (Chi-cuadrado con alfa=0.05), traiciona siempre.
    - Si el oponente parece ser TFT o un clon, juega TFT.
    - En otro caso, coopera y realiza traiciones aleatorias cada 5 a 15 movimientos.
    """

    def __init__(self):
        super().__init__()
        self.ronda = 0
        self.alfa = 0.05

        self.oponente_es_random = False
        self.oponente_es_clon = True
        self.oponente_es_TFT = True

        self.historial_oponente = deque(maxlen=2)
        self.historial_propio = deque(maxlen=2)
        self.ultimo_movimiento_oponente = Elecciones.COOPERAR
        self.ultimo_movimiento_propio = Elecciones.COOPERAR
        self.siguiente_turno_traicion_aleatoria = None

        self.traiciones_oponente = 0
        self.cooperaciones_oponente = 0

    def realizar_eleccion(self) -> Elecciones:
        """
        Lógica de decisión según las fases y detecciones descritas.
        """
        # Fase inicial: TFT 50 rondas
        if self.ronda <= 50:
            self._chequear_tipo_oponente()
            decision = Elecciones.COOPERAR if self.ronda == 0 else self.ultimo_movimiento_oponente
            self.ultimo_movimiento_propio = decision
            self.historial_propio.append(decision)
            return decision

        # Ronda 51: traiciona
        if self.ronda == 51:
            self.ultimo_movimiento_propio = Elecciones.TRAICIONAR
            self.historial_propio.append(Elecciones.TRAICIONAR)
            return Elecciones.TRAICIONAR

        # Siguientes 5 rondas: TFT (52..56)
        if 52 <= self.ronda <= 56:
            decision = self.ultimo_movimiento_oponente
            self.ultimo_movimiento_propio = decision
            self.historial_propio.append(decision)
            return decision

        # Evaluar aleatoriedad del oponente
        _, p_value = chisquare([self.cooperaciones_oponente, self.traiciones_oponente])
        self.oponente_es_random = p_value >= self.alfa

        if self.oponente_es_random:
            self.ultimo_movimiento_propio = Elecciones.TRAICIONAR
            self.historial_propio.append(Elecciones.TRAICIONAR)
            return Elecciones.TRAICIONAR

        # Si parece clon o TFT, jugar TFT
        if self.oponente_es_clon or self.oponente_es_TFT:
            decision = self.ultimo_movimiento_oponente
            self.ultimo_movimiento_propio = decision
            self.historial_propio.append(decision)
            return decision

        # En otro caso: cooperar con traiciones aleatorias cada 5-15 movimientos
        if self.siguiente_turno_traicion_aleatoria is None:
            self.siguiente_turno_traicion_aleatoria = self.ronda + random.randint(5, 15)

        if self.ronda == self.siguiente_turno_traicion_aleatoria:
            # remuestrear próximo turno de traición
            self.siguiente_turno_traicion_aleatoria = self.ronda + random.randint(5, 15)
            self.ultimo_movimiento_propio = Elecciones.TRAICIONAR
            self.historial_propio.append(Elecciones.TRAICIONAR)
            return Elecciones.TRAICIONAR

        self.ultimo_movimiento_propio = Elecciones.COOPERAR
        self.historial_propio.append(Elecciones.COOPERAR)
        return Elecciones.COOPERAR

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones):
        """
        Actualiza estado con la acción del oponente y aumenta el contador de rondas.
        """
        self.ronda += 1
        self.ultimo_movimiento_oponente = eleccion
        self.historial_oponente.append(eleccion)
        if eleccion == Elecciones.TRAICIONAR:
            self.traiciones_oponente += 1
        else:
            self.cooperaciones_oponente += 1
        self._chequear_tipo_oponente()

    def notificar_nuevo_oponente(self) -> None:
        """
        Reset de estado para un nuevo oponente.
        """
        self.ronda = 0
        self.oponente_es_random = False
        self.oponente_es_clon = True
        self.oponente_es_TFT = True
        self.historial_oponente.clear()
        self.historial_propio.clear()
        self.ultimo_movimiento_oponente = Elecciones.COOPERAR
        self.ultimo_movimiento_propio = Elecciones.COOPERAR
        self.siguiente_turno_traicion_aleatoria = None
        self.traiciones_oponente = 0
        self.cooperaciones_oponente = 0

    def _chequear_tipo_oponente(self) -> bool:
        """
        Detecta si el oponente es clon (juega exactamente lo mismo)
        y si es TFT (responde con la acción propia de la ronda previa).
        Retorna True para conveniencia en tests.
        """
        # Clon: compara últimos movimientos propios vs del oponente cuando hay historial
        if len(self.historial_oponente) == len(self.historial_propio) and len(self.historial_oponente) > 0:
            if any(o != p for o, p in zip(self.historial_oponente, self.historial_propio)):
                self.oponente_es_clon = False

        # TFT: el oponente repite nuestra acción previa
        if len(self.historial_propio) >= 1 and self.ultimo_movimiento_oponente != self.historial_propio[-1]:
            self.oponente_es_TFT = False

        return True


