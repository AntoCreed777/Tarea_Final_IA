from src.selectores_de_oponentes.base_selector_de_oponentes import \
    BaseSelectorDeOponentes
from src.strategies import base_strategies


class SelectorPvP(BaseSelectorDeOponentes):
    """
    Selector de oponentes que organiza enfrentamientos usando un calendario
    round-robin en formato de rondas (matchdays).

    Este selector garantiza que:

    - Cada estrategia se enfrenta exactamente una vez contra todas las demás.
    - Dentro de cada ronda, ninguna estrategia se repite, lo que permite la
      ejecución segura en entornos multihilo (cada instancia participa solo
      en un duelo por ronda).
    - El ciclo completo de rondas cubre todas las combinaciones posibles de
      duelos.
    """

    def __init__(self):
        """
        Inicializa el selector de oponentes.

        Atributos
        ---------
        rondas : list[list[tuple]]
            Lista de rondas generadas. Cada ronda contiene una lista de pares
            (e1, e2), donde e1 y e2 son instancias de estrategias que deben
            enfrentarse.
        ronda_actual : int
            Índice de la ronda actualmente en ejecución.
        """
        self.rondas: list[list[base_strategies]] = []
        self.ronda_actual = 0

    def seleccionar(self, estrategias):
        """
        Selecciona un par de estrategias para ser enfrentadas según el
        calendario round-robin generado.

        Parámetros
        ----------
        estrategias : list
            Lista de instancias de estrategias activas en el torneo.

        Retorna
        -------
        tuple
            Un par `(e1, e2)` correspondiente al siguiente duelo a ejecutar.
        bool
            `True` si la ronda actual terminó luego de extraer el duelo,
            `False` en caso contrario.

        Notas
        -----
        - Si se completan todas las rondas, el calendario se reinicia
          automáticamente.
        """
        # Crear calendario si no existe
        if not self.rondas:
            self._generar_rondas(estrategias)

        # Reinicio de calendario si se completó
        if self.ronda_actual >= len(self.rondas):
            self.ronda_actual = 0

        ronda = self.rondas[self.ronda_actual]

        # Avanzar a la siguiente ronda si la actual está vacía
        if not ronda:
            self.ronda_actual += 1
            return self.seleccionar(estrategias)

        # Extraer un duelo de la ronda actual
        duelo = ronda.pop()

        # Detectar si la ronda terminó luego del pop()
        ciclo_terminado = not ronda

        return duelo, ciclo_terminado

    def _generar_rondas(self, estrategias):
        """
        Genera un calendario round-robin completo mediante el algoritmo
        conocido como "Método del Círculo".

        Este algoritmo asegura que:

        - Cada participante enfrente a todos los demás.
        - En cada ronda, todos los participantes quedan emparejados una única vez.
        - Si el número de estrategias es impar, se agrega un marcador nulo
          (None) que actúa como "bye" o descanso, evitando excepciones.

        Parámetros
        ----------
        estrategias : list
            Lista original de estrategias participantes.

        Detalles del Algoritmo
        -----------------------
        1. Si el número de estrategias es impar, se agrega un `None`.
        2. El conjunto se divide en dos mitades: izquierda y derecha.
        3. En cada ronda:
           - Se empareja cada elemento de la izquierda con el correspondiente
             de la derecha.
           - Se rota la lista manteniendo fijo el primer elemento, lo que
             produce nuevas combinaciones sin repeticiones.
        4. Se generan exactamente `n - 1` rondas para `n` participantes.
        """
        estrategias = estrategias[:]  # Copiar para evitar efectos externos
        n = len(estrategias)

        # Si N es impar, se agrega un "bye"
        if n % 2 == 1:
            estrategias.append(None)
            n += 1

        mitad = n // 2
        izquierda = estrategias[:mitad]
        derecha = estrategias[mitad:][::-1]

        self.rondas = []

        for _ in range(n - 1):
            ronda = []

            # Emparejamiento de la ronda actual
            for e1, e2 in zip(izquierda, derecha):
                if e1 is not None and e2 is not None:
                    ronda.append((e1, e2))

            self.rondas.append(ronda)

            # Rotación circular según el método round-robin
            ultimo_izq = izquierda.pop()
            primero_der = derecha.pop(0)
            izquierda.insert(1, primero_der)
            derecha.append(ultimo_izq)
