import random

import pandas as pd
import enum

from src.controlador_duelos import ControladorDuelos
from src.selectores_de_oponentes import SelectorAllForOne
from src.selectores_de_oponentes.selector_pvp import SelectorPvP
from src.strategies import *
from src.strategies.RL.Estados import StatState, HistoryState, HistoryStatState
from src.strategies.RL.politicas import EpsilonGreedy

# ejecutar los algoritmos en paralelo, y cada cierta cantidad de rondas 
# guardar la funcion de perdida, y el porcentaje exploracion


class Metrica(enum.Enum):
    PERDIDA = 0
    EXPLORACION = 1

# ejecutar torneos de entrenamiento para una estrategia dada, y guardar las metricas solicitadas
def torneos_de_entrenamiento(estrategia, metrica_a_guardar, episodios_por_torneo=100, cantidad_de_torneos=100, guardado_cada_n_torneos=10):

    estategias_a_enfrentar = [
        TitForTat(),
        SiempreCoopera(),
        SiempreTraiciona(),
        Random(),
        Davis(),
        Downing(),
        Feld(),
        Grofman(),
        Joss(),
        Shubik(),
        SteinRapoport(),
        TidemanChieruzzi(),
        Tullock(),
        Nydegger(),
        Graaskamp(),
        Grudger(),
        Anonymous()
    ]

    # quitar estrategia de la lista si ya esta
    if estrategia.__class__.__name__ in estategias_a_enfrentar:
        estategias_a_enfrentar.remove(estrategia.__class__.__name__)

    df = pd.DataFrame(columns=["estrategia", "n_torneo", "perdida", "exploracion", "puntaje_torneo", "puntaje_acumulado"])

    puntaje_acumulado = 0

    # ejecutar los torneos 1 contra todos
    for torneo in range(cantidad_de_torneos):

        print(f"Iniciando torneo {torneo + 1}/{cantidad_de_torneos} para {estrategia.__class__.__name__}")

        controlador = ControladorDuelos(
            estrategias_a_enfrentar=estategias_a_enfrentar,
            cantidad_de_torneos=1,
            jugadas_base_duelo=episodios_por_torneo,
            limite_de_variacion_de_jugadas=0,
            selector_de_oponentes=SelectorAllForOne(estrategia)
        )
        controlador.iniciar_duelos(analisis=False, verbose=True)
        puntaje_acumulado += estrategia.puntaje_torneo_actual

        if (torneo + 1) % guardado_cada_n_torneos == 0:
            fila = {"estrategia": estrategia.__class__.__name__,
                    "n_torneo": torneo + 1,
                    "puntaje_torneo": estrategia.puntaje_torneo_actual,
                    "puntaje_acumulado": puntaje_acumulado
                    }
            
            # Obtener la métrica solicitada; si no existe, no agregar fila
            try:
                if Metrica.PERDIDA == metrica_a_guardar:
                    fila["perdida"] = estrategia.get_loss()
                elif Metrica.EXPLORACION == metrica_a_guardar:
                    fila["exploracion"] = estrategia.get_exploration()
            except AttributeError:
                # La estrategia no tiene la métrica solicitada: saltar agregación
                print("La estrategia no tiene la métrica solicitada.")
                fila = None

            # Excluir entradas None/NA para mantener el comportamiento antiguo y evitar FutureWarning
            if fila is not None:
                fila_filtrada = {k: v for k, v in fila.items() if v is not None}
                if (
                    (metrica_a_guardar == Metrica.PERDIDA and "perdida" in fila_filtrada) or
                    (metrica_a_guardar == Metrica.EXPLORACION and "exploracion" in fila_filtrada)
                ):
                    df = pd.concat([df, pd.DataFrame([fila_filtrada])], ignore_index=True)

    return df

if __name__ == "__main__":
    #random.seed(42)

    cantidad_de_torneos = 5
    jugadas_base_duelo = 100
    limite_de_variacion_de_jugadas = 10

    # estrategias = [
    #     TitForTat(),
    #     SiempreCoopera(),
    #     SiempreTraiciona(),
    #     Random(),
    #     Davis(),
    #     Downing(),
    #     Feld(),
    #     Grofman(),
    #     Joss(),
    # ]

    protas = [
        [DeepQNetwork(
            tamaño_estado=20,
            alpha=0.2,
            gamma=float(1 - (1 / jugadas_base_duelo)),
            use_opponent_context=True,
        ), Metrica.PERDIDA],
        [A2C(), Metrica.PERDIDA],
        [DuelingDQN(), Metrica.PERDIDA],
        [A2C_LSTM(), Metrica.PERDIDA],

    ]

    dataf = pd.DataFrame()

    for prota, metrica in protas:
        # print(type(prota))
        # enemigos = estrategias.copy()
        # enemigos.append(prota) #Para visualizar su puntaje, despues se puede mejorar
        # torneo = ControladorDuelos(
        #         enemigos,
        #         cantidad_de_torneos,
        #         jugadas_base_duelo,
        #         limite_de_variacion_de_jugadas,
        #         selector_de_oponentes=SelectorAllForOne(prota),
        # )
        # torneo.iniciar_duelos()

        resultado = torneos_de_entrenamiento(
            prota,
            metrica, 
            cantidad_de_torneos=cantidad_de_torneos,
            guardado_cada_n_torneos=1
        )
            
        print(resultado)

        dataf = pd.concat([dataf, resultado], ignore_index=True)

       # prota.export_QTable(f"{prota.__class__.__name__}+TFT2")


    print(dataf)