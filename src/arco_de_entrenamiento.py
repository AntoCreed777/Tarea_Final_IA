import random
from multiprocessing import Pool, cpu_count

import pandas as pd
import enum


from tqdm import tqdm

from src.controlador_duelos import ControladorDuelos
from src.selectores_de_oponentes import SelectorAllForOne
from src.selectores_de_oponentes.selector_pvp import SelectorPvP
from src.strategies import *
from src.strategies.RL.Estados import StatState, HistoryState, HistoryStatState
from src.strategies.RL.politicas import EpsilonGreedy
from src.strategies.RL.politicas.softmax import Softmax


# ejecutar los algoritmos en paralelo, y cada cierta cantidad de rondas
# guardar la funcion de perdida, y el porcentaje exploracion


class Metrica(enum.Enum):
    PERDIDA = 0,
    EXPLORACION = 1

def crear_agente(str):
    agente = None
    if str == "DeepQNetwork":
       agente = DeepQNetwork(tamaño_estado=20,
                alpha= 1e-3 ,
                gamma= 0.97 ,
                start_epsilon= 0.2,
                end_epsilon= 0.02 ,
                rounds_of_decay_epsilon= 1500,
                replay_capacity=10000,
                batch_size= 128,
                num_hidden_layers=2,
                use_opponent_context= True,
                context_window= 10)
    elif str == "A2C":
        agente = A2C(lr= 1e-3,
                    gamma=0.95,
                    entropy_coef= 5e-3,
                    value_coef=0.5,
                    tamaño_estado= 20
                    )
    elif str == "Dueling_dqn":
        agente = DuelingDQN(lr=5e-4,
                            gamma= 0.98,
                            replay_capacity= 20000,
                            batch_size= 128,
                            tau= 0.005,
                            hidden=128,
                            use_opponent_context= True,
                            context_window=10
                            )
    elif str == "LSTM":
        agente = A2C_LSTM(tamaño_estado= 20,
                          lr= 5e-4,
                          gamma= 0.96,
                          entropy_coef=3e-3,
                          value_coef= 0.5,
                          reset_history_on_new_opponent = False
                          )
    elif str == "SARSA":
        agente = SARSA(Softmax(2.5),HistoryStatState(n_grupos=3,short_memory=10,tamaño_estado=2),
                       alpha=0.1, gamma=0.999)
    else:
        agente = QLearning(EpsilonGreedy(start_epsilon= 0.9, end_epsilon= 0.3, rounds_of_decay_epsilon=800000000),
                           HistoryStatState(n_grupos=3,short_memory=15,tamaño_estado=2),alpha=0.1, gamma=0.999)
    return agente



# ejecutar torneos de entrenamiento para una estrategia dada, y guardar las metricas solicitadas
def torneos_de_entrenamiento(args) -> pd.DataFrame:
    estrategia, metrica_a_guardar, seed, episodios_por_torneo, limite_de_variacion_de_jugadas, cantidad_repeticiones_de_torneos, guardado_cada_n_torneos = args
    # Permitir pasar una especificación serializable de la estrategia
    # en forma de (ClaseEstrategia, kwargs) para evitar problemas de pickling.
    estrategia = crear_agente(estrategia)
    estrategias_a_enfrentar = [
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

    random.seed(seed)

    estrategias = estrategias_a_enfrentar.copy()
    estrategias.append(estrategia)

    df = pd.DataFrame(columns=["estrategia", "n_torneo", "perdida", "exploracion", "puntaje_torneo", "puntaje_acumulado", "seed"])


    # ejecutar los torneos 1 contra todos
    for torneo in range(cantidad_repeticiones_de_torneos):

        controlador = ControladorDuelos(
            estrategias_a_enfrentar=estrategias,
            cantidad_de_torneos=1,
            jugadas_base_duelo=episodios_por_torneo,
            limite_de_variacion_de_jugadas=limite_de_variacion_de_jugadas,
            selector_de_oponentes=SelectorAllForOne(estrategia)
        )

        analisis = controlador.iniciar_duelos(analisis=True, verbose=False)

        # Guardar las métricas cada n torneos
        if (torneo + 1) % guardado_cada_n_torneos == 0:
            fila = {"estrategia": estrategia.__class__.__name__,
                    "n_torneo": torneo + 1,
                    "puntaje_torneo": estrategia.puntaje_torneo_actual,
                    "puntaje_acumulado": analisis[estrategia.__class__.__name__],
                    "seed": seed
                    }
            
            # Obtener la métrica solicitada; si no existe, no agregar fila
            try:
                if Metrica.PERDIDA == metrica_a_guardar:
                    fila["perdida"] = estrategia.get_loss()
                elif Metrica.EXPLORACION == metrica_a_guardar:
                    fila["exploracion"] = estrategia.porcentaje_explorado()
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
    estrategia.save(f"{estrategia.__class__.__name__}{seed}")
    return df

if __name__ == "__main__":
    #random.seed(42)

    cantidad_repeticiones_de_torneos = 1     ## cuantas tandas 
    cantidad_de_torneos = 1                  ## cuantos torneos por tanda
    guardado_cada_n_torneos = 1              ## cada cuantos torneos guardar las metricas
    jugadas_base_duelo = 10                  ## cantidad de jugadas base por duelo
    limite_de_variacion_de_jugadas = 10      ## limite de variacion aleatoria en la cantidad de jugadas por duelo


    protas = [["DeepQNetwork",Metrica.PERDIDA],
    ]

    dataf = pd.DataFrame()
    seeds = [1 ]
    tasks = []
    for prota in protas:
        for seed in seeds:
            if prota[1] == Metrica.PERDIDA:
                tasks.append((prota[0], prota[1] ,seed,200, 10 ,40, 1))
            else:
                tasks.append((prota[0], prota[1] ,seed,1000, 50, 100000, 1000))

    print(f"Ejecutando {len(tasks)} experimentos en paralelo...")

    # Usa todos los cores disponibles menos 1
    n_workers = max(cpu_count() - 2, 1)
    print(f"Usando {n_workers} procesos")

    # rows = []
    with Pool(n_workers) as pool:
         for result in tqdm(pool.imap_unordered(torneos_de_entrenamiento, tasks), total=len(tasks)):
             dataf = pd.concat([dataf, result], ignore_index=True)
    dataf.to_csv("resultados_arco_de_entrenamiento_CAMBIAR_NOMBRE.csv", index=False)