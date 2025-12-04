import queue
from collections import deque
from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Nydegger(base_strategies):
    """
    Estrategia Nydegger:
    - Usa Tit for Tat (TFT) en los primeros 3 movimientos, con la excepción de que
      si en el primer movimiento fue la única en cooperar (CT) y en el segundo la
      única en traicionar (TC), entonces traiciona en el tercer movimiento.
    - A partir del cuarto movimiento, la elección depende del valor A:
      A se calcula sumando, para cada una de las últimas 3 rondas:
        * 2 puntos si el oponente traicionó,
        * 1 punto si la estrategia traicionó,
        * 0 en caso contrario.
      Estos puntos se ponderan con pesos 16, 4 y 1 respectivamente,
      desde la más reciente a la más antigua (orden cronológico inverso).
    - Traiciona solamente si A pertenece al conjunto:
      {1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61}.

    Notas:
      - Codificación por ronda usada internamente:
        CC -> 0, TC -> 1 (yo traicioné), CT -> 2 (oponente traicionó), TT -> 3 (ambos traicionaron).
      - 'puntaje' es un acumulador de recompensas definido en la clase base.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.

    Atributos:
        ultima_respuesta_oponente (Elecciones):
            Última acción tomada por el oponente. Se inicializa en Elecciones.COOPERAR
            y se actualiza al final de cada ronda.
        ultima_respuesta_propia (Elecciones):
            Última acción tomada por esta estrategia. Se actualiza en realizar_eleccion()
            antes de ser registrada en el historial.
        historial_respuestas (collections.deque[int]):
            Estructura con maxlen=3 que almacena las últimas tres rondas codificadas como:
            0 (CC), 1 (TC: yo traicioné), 2 (CT: oponente traicionó), 3 (TT: ambos traicionaron).
    """

    def __init__(self):
        """
        Inicializa el estado interno asumiendo cooperación inicial del oponente.
        """
        super().__init__()

        self.ultima_respuesta_oponente = Elecciones.COOPERAR
        self.ultima_respuesta_propia = Elecciones.COOPERAR

        # Historial de los últimos 3 resultados (codificados: 0/1/2/3) con límite fijo.
        self.historial_respuestas = deque(maxlen=3)

    def realizar_eleccion(self) -> Elecciones:
        """
        Decide la siguiente acción según las reglas de Nydegger.

        Returns:
            Elecciones: La acción a tomar (COOPERAR o TRAICIONAR).
        """

        # Primeros 3 movimientos: TFT con excepción en el tercero descrita arriba.
        if len(self.historial_respuestas) < 3:

            # Excepción: si el historial es [CT, TC] -> [2, 1], traiciona en el tercero.
            if len(self.historial_respuestas) == 2:
                if self.historial_respuestas[0] == 2 and self.historial_respuestas[1] == 1:
                    self.ultima_respuesta_propia = Elecciones.TRAICIONAR
                    return Elecciones.TRAICIONAR

            # Por defecto: Tit for Tat (replicar la última respuesta del oponente).
            self.ultima_respuesta_propia = self.ultima_respuesta_oponente
            return self.ultima_respuesta_oponente
        
        else: 
            # Calcular A usando los últimos 3 resultados con pesos 16 (más reciente), 4 y 1.
            # Cada resultado aporta: 2 si oponente traicionó, 1 si la estrategia traicionó, 0 si ambos cooperaron.
            pesos = [16, 4, 1]  # más reciente, anterior, tercero anterior
            A = 0

            # iterar del más reciente al más antiguo
            for idx, resultado in enumerate(reversed(self.historial_respuestas)):
                A += resultado * pesos[idx]

            # conjunto donde se traiciona
            defectos = {1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61}

            if A in defectos:
                self.ultima_respuesta_propia = Elecciones.TRAICIONAR
                return Elecciones.TRAICIONAR
            else:
                self.ultima_respuesta_propia = Elecciones.COOPERAR
                return Elecciones.COOPERAR

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones):
        """
        Registra el resultado de la ronda actual en el historial codificado:
          - CC: 0
          - TC: 1 (yo traicioné, oponente cooperó)
          - CT: 2 (yo cooperé, oponente traicionó)
          - TT: 3 (ambos traicionaron)

        Args:
            eleccion (Elecciones): Acción tomada por el oponente.
        """
        self.ultima_respuesta_oponente = eleccion

        # registrar el resultado de la ronda actual en formato numérico
        if self.ultima_respuesta_propia == Elecciones.COOPERAR and self.ultima_respuesta_oponente == Elecciones.COOPERAR:
            self.historial_respuestas.append(0) 

        elif self.ultima_respuesta_propia == Elecciones.TRAICIONAR and self.ultima_respuesta_oponente == Elecciones.COOPERAR:
            self.historial_respuestas.append(1)  # estrategia traiciono

        elif self.ultima_respuesta_propia == Elecciones.COOPERAR and self.ultima_respuesta_oponente == Elecciones.TRAICIONAR:
            self.historial_respuestas.append(2)  # oponente traicionó
       
        else:
            self.historial_respuestas.append(3) # 2 del oponente + 1 propio

    def notificar_nuevo_oponente(self) -> None:
        """
        Restablece el estado interno para un nuevo oponente:
        - Asume cooperación inicial.
        - Limpia el historial de resultados.
        """
        self.ultima_respuesta_oponente = Elecciones.COOPERAR
        self.ultima_respuesta_propia = Elecciones.COOPERAR
        self.historial_respuestas.clear()
