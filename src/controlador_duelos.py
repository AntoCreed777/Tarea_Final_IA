import random
from concurrent.futures import ThreadPoolExecutor

from src.elecciones import Elecciones
from src.selectores_de_oponentes import BaseSelectorDeOponentes
from src.strategies import base_strategies


class ControladorDuelos:
    """
    Controlador encargado de ejecutar múltiples torneos y administrar los duelos
    entre estrategias del dilema del prisionero iterado.

    Esta clase coordina la asignación aleatoria de enfrentamientos, la cantidad
    variable de jugadas por duelo, la notificación de nuevos oponentes y la
    asignación de recompensas según las reglas definidas.

    Atributos:
        estrategias_a_enfrentar (list[base_strategies]):
            Conjunto de estrategias participantes en los torneos.
        cantidad_de_torneos (int):
            Número total de torneos que se ejecutarán.
        jugadas_base_duelo (int):
            Cantidad base de rondas por duelo antes de aplicar variaciones.
        limite_de_variacion_de_jugadas (int):
            Valor máximo de variación positiva o negativa sobre la cantidad base
            de jugadas por duelo.
    """

    def __init__(
        self,
        estrategias_a_enfrentar: list[base_strategies],
        cantidad_de_torneos: int,
        jugadas_base_duelo: int,
        limite_de_variacion_de_jugadas: int,
        selector_de_oponentes: BaseSelectorDeOponentes,
    ):
        """
        Inicializa el controlador validando los parámetros principales del
        torneo y almacenando las estrategias que participarán.

        Args:
            estrategias_a_enfrentar (list[base_strategies]):
                Lista de estrategias que participarán en todos los torneos.
            cantidad_de_torneos (int):
                Cantidad total de torneos a realizar.
            jugadas_base_duelo (int):
                Número base de jugadas por duelo.
            limite_de_variacion_de_jugadas (int):
                Variación máxima (positiva o negativa) respecto al número base.

        Raises:
            ValueError: Si la cantidad de estrategias es menor a 2.
            ValueError: Si la cantidad de torneos es menor o igual a 0.
            ValueError: Si el número base de jugadas es menor a 5.
            ValueError: Si la variación de jugadas es mayor o igual al número base.
        """
        if len(estrategias_a_enfrentar) < 2:
            raise ValueError("Deben de haber como minimo 2 estrategias para enfrentar.")

        self.estrategias_a_enfrentar = estrategias_a_enfrentar

        if cantidad_de_torneos <= 0:
            raise ValueError("La cantidad de torneos debe de ser igual o mayor a 1.")

        self.cantidad_de_torneos = cantidad_de_torneos

        if jugadas_base_duelo < 5:
            raise ValueError(
                "La cantidad de jugadas base debe ser un número mayor o igual a 5."
            )

        self.jugadas_base_duelo = jugadas_base_duelo

        if limite_de_variacion_de_jugadas >= jugadas_base_duelo:
            raise ValueError(
                "El límite de variación de jugadas no puede ser igual o superior "
                "a la cantidad de jugadas base."
            )

        self.limite_de_variacion_de_jugadas = limite_de_variacion_de_jugadas
        self.selector_de_oponentes = selector_de_oponentes

    def iniciar_duelos(self):
        """
        Ejecuta todos los torneos configurados.

        Para cada torneo:
        - Se clona la lista de estrategias participantes.
        - Se seleccionan pares aleatorios de estrategias para enfrentarse.
        - Cada estrategia es notificada del inicio de un nuevo enfrentamiento.
        - La cantidad de jugadas del duelo se determina aplicando variación aleatoria.
        - En cada jugada, ambas estrategias toman decisiones, reciben información
          sobre la decisión del oponente y luego se asignan recompensas.

        El proceso continúa hasta que no queden estrategias sin enfrentar dentro
        del torneo.
        """
        for i in range(self.cantidad_de_torneos):
            print(f"\033[1;97m{'-' * 10} Torneo {i+1} iniciado {'-' * 10}\033[0m")

            aux_estrategias_a_enfrentar = self.estrategias_a_enfrentar.copy()
            duelos = []

            (e1, e2), termino = self.selector_de_oponentes.seleccionar(
                aux_estrategias_a_enfrentar
            )

            self._preparar_duelo(e1, e2, duelos)

            while not termino:
                (e1, e2), termino = self.selector_de_oponentes.seleccionar(
                    aux_estrategias_a_enfrentar
                )
                self._preparar_duelo(e1, e2, duelos)

            # Paralelizar los duelos
            with ThreadPoolExecutor() as executor:
                """
                Ejecuta todos los duelos en paralelo usando hilos.

                Cada duelo se pasa como argumento a la función `duelo`.
                ThreadPoolExecutor se encarga de crear hilos y esperar que
                todos los duelos terminen antes de continuar con el siguiente torneo.
                """
                executor.map(lambda args: self.duelo(*args), duelos)

            self._mostrar_puntajes()

            for e in self.estrategias_a_enfrentar:
                e.notificar_nuevo_torneo()

    def _mostrar_puntajes(self):
        print("\n", "-" * 5, "Puntajes Acumulados", "-" * 5)
        for e in sorted(
            self.estrategias_a_enfrentar, key=lambda est: est.puntaje, reverse=True
        ):
            print(e.get_puntaje_acumulado())

        print("\n", "-" * 5, "Puntajes del Torneo Actual", "-" * 5)
        for e in sorted(
            self.estrategias_a_enfrentar,
            key=lambda est: est.puntaje_torneo_actual,
            reverse=True,
        ):
            print(e.get_puntaje_de_este_torneo())

        print("\n\n")

    def duelo(
        self,
        estrategia1: base_strategies,
        estrategia2: base_strategies,
        cantidad_jugadas: int,
    ):
        """
        Ejecuta un duelo entre dos estrategias durante un número determinado de jugadas.

        Para cada jugada:
        - Cada estrategia realiza su elección (Cooperar o Traicionar).
        - Cada estrategia recibe la elección del oponente.
        - Se asignan recompensas según las reglas del dilema del prisionero iterado.

        Args:
            estrategia1 (base_strategies): Primera estrategia participante en el duelo.
            estrategia2 (base_strategies): Segunda estrategia participante en el duelo.
            cantidad_jugadas (int): Número total de rondas que se jugarán en este duelo.
        """
        first, second = sorted([estrategia1, estrategia2], key = id)

        first._jugando.acquire()
        second._jugando.acquire()

        estrategia1.notificar_nuevo_oponente()
        estrategia2.notificar_nuevo_oponente()

        for _ in range(cantidad_jugadas):
            eleccion1 = estrategia1.realizar_eleccion()
            eleccion2 = estrategia2.realizar_eleccion()

            estrategia1.recibir_eleccion_del_oponente(eleccion2)
            estrategia2.recibir_eleccion_del_oponente(eleccion1)

            self.otorgar_recompensas(estrategia1, estrategia2, eleccion1, eleccion2)

        first._jugando.release()
        second._jugando.release()

    def _preparar_duelo(
        self,
        e1: base_strategies,
        e2: base_strategies,
        duelos: list[tuple[base_strategies, base_strategies, int]],
    ):
        # Determinar número de jugadas del duelo
        cantidad_jugadas = self.jugadas_base_duelo + random.randint(
            -self.limite_de_variacion_de_jugadas,
            self.limite_de_variacion_de_jugadas,
        )

        duelos.append((e1, e2, cantidad_jugadas))

    def otorgar_recompensas(self, e1, e2, c1, c2):
        """
        Asigna recompensas a ambas estrategias según las reglas clásicas del
        dilema del prisionero iterado.

        Reglas aplicadas:
            - Cooperar / Cooperar → +3 para ambos
            - Cooperar / Traicionar → +0 para el primero, +5 para el segundo
            - Traicionar / Cooperar → +5 para el primero, +0 para el segundo
            - Traicionar / Traicionar → +1 para ambos

        Args:
            e1, e2: Estrategias participantes del duelo.
            c1, c2 (Elecciones): Elecciones tomadas por cada estrategia.
        """
        if c1 == Elecciones.COOPERAR and c2 == Elecciones.COOPERAR:
            e1.recibir_recompensa(3)
            e2.recibir_recompensa(3)

        elif c1 == Elecciones.COOPERAR and c2 == Elecciones.TRAICIONAR:
            e1.recibir_recompensa(0)
            e2.recibir_recompensa(5)

        elif c1 == Elecciones.TRAICIONAR and c2 == Elecciones.COOPERAR:
            e1.recibir_recompensa(5)
            e2.recibir_recompensa(0)

        else:
            e1.recibir_recompensa(1)
            e2.recibir_recompensa(1)
