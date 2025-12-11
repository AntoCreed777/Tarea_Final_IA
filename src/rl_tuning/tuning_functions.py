import itertools
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import sys

from src.strategies.RL.politicas.softmax import Softmax

sys.path.append(r"/")

from src.controlador_duelos import ControladorDuelos
from src.selectores_de_oponentes import SelectorAllForOne
from src.strategies import SARSA, QLearning, TitForTat, Davis, Downing, Feld, Grofman, SiempreCoopera, SiempreTraiciona
from src.strategies.RL.Estados import StatState, HistoryState, HistoryStatState
from src.strategies.RL.politicas import EpsilonGreedy


def crear_agente(config):
    sm = None
    if config["state_manager"] == "history":
        sm = HistoryState(tamaño_estado=config["historial"])
    elif config["state_manager"] == "stats_summary":
        sm = StatState(grupos=config["grupos"], short_memory=config["memory"])
    else:
        sm = HistoryStatState(n_grupos=config["grupos"], short_memory=config["memory"], tamaño_estado=config["historial"])

    agente = None
    if config["algo"] == "SARSA":
        agente = SARSA(EpsilonGreedy(start_epsilon=0.7, end_epsilon=0.5, rounds_of_decay_epsilon=900000),
                       sm, alpha=config["alpha"], gamma=(1 - (1 / 100)))
    else:
        agente = QLearning(EpsilonGreedy(start_epsilon=0.7, end_epsilon=0.5, rounds_of_decay_epsilon=900000),
                           sm, alpha=config["alpha"], gamma=(1 - (1 / 100)))
    return agente


def train_one_run(agente, seed, episodes):
    random.seed(seed);
    np.random.seed(seed)

    estrategias = [
        TitForTat(),
        Davis(),
        Downing(),
        Feld(),
        SiempreTraiciona(),
        agente
    ]
    torneo = ControladorDuelos(estrategias, episodes, 100, 0, selector_de_oponentes=SelectorAllForOne(agente))

    analisis = torneo.iniciar_duelos(analisis=True, verbose=False)
    return agente, analisis[agente.__class__.__name__]


def eval_policy(agent, n_episodes, seed_start=0):
    """
    Evalúa agent sin exploración por n_episodes.
    Devuelve: list de returns por episodio y métricas del dominio (ej. coop_rate)
    """
    agent.freeze()
    estrategias = [
        TitForTat(),
        Davis(),
        Downing(),
        Feld(),
        SiempreTraiciona(),
        agent
    ]

    torneo = ControladorDuelos(estrategias, n_episodes, 100, 0, selector_de_oponentes=SelectorAllForOne(agent))
    analisis = torneo.iniciar_duelos(analisis=True, verbose=False)
    return analisis[agent.__class__.__name__]


# -------------------------
# Espacio de búsqueda (ejemplo)
# -------------------------
def generate_configs():
    state_managers = ["history", "stats_summary", "mix"]
    historial = [3, 4, 5]
    memory = [5, 10, 15]
    grupos = [3, 4, 5]
    algos = ["Q", "SARSA"]
    alphas = [0.01, 0.05, 0.1]
    historial_corto = [1, 2, 3]
    id = 1
    configs = []

    for sm in state_managers:
        if sm == "history":
            for h in historial:
                for algo in algos:
                    for a in alphas:
                        configs.append({
                            "historial": h, "memory": None, "grupos": None,
                            "state_manager": sm, "algo": algo,
                            "alpha": a, "id" : id
                        })
                        id += 1
        else:
            for m in memory:
                for group in grupos:
                    for algo in algos:
                        for a in alphas:
                            if sm == "mix":
                                for h in historial_corto:
                                    configs.append({
                                        "memory": m, "grupos": group, "historial": h,
                                        "state_manager": sm, "algo": algo, "alpha": a, "id" : id
                                    })
                                    id +=1
                            else:
                                configs.append({
                                    "memory": m, "grupos": group, "historial": None,
                                    "state_manager": sm, "algo": algo, "alpha": a , "id" : id
                                })
                                id +=1
    return configs


# -------------------------
# Runner experimental
# -------------------------
def run_experiments(configs, seeds, episodes_train=2000, n_eval=50):
    rows = []
    for cfg in tqdm(configs, desc="configs"):
        for seed in seeds:
            try:
                agent, train_rewards = train_one_run(cfg, seed, episodes_train)
            except Exception as e:
                print("Error en config", cfg, "seed", seed, e)
                continue

            eval_returns = eval_policy(agent, n_eval, seed_start=seed * 1000)

            row = {
                **cfg,
                "seed": seed,
                "train_mean": float(np.mean(train_rewards)),
                "train_std": float(np.std(train_rewards)),
                "eval_mean": float(np.mean(eval_returns)),
                "eval_std": float(np.std(eval_returns)),
                "eval_median": float(np.median(eval_returns)),
                "eval_p25": float(np.percentile(eval_returns, 25)),
                "eval_p75": float(np.percentile(eval_returns, 75)),
                "p_explorado": agent.porcentaje_explorado()
            }

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def evaluar_crecimiento(args):
    config, seed, episodes_train, n_eval = args

    print(f"[worker] Iniciando experimento cfg={config['id']} seed={seed}")
    sys.stdout.flush()

    random.seed(seed);
    np.random.seed(seed)

    agente = crear_agente(config)
    estrategias = [
        TitForTat(),
        Davis(),
        Downing(),
        Feld(),
        SiempreTraiciona(),
        agente
    ]
    fila = {**config}

    torneo_entrenar = ControladorDuelos(estrategias, episodes_train, 100, 0, selector_de_oponentes=SelectorAllForOne(agente))
    torneo_probar = ControladorDuelos(estrategias, n_eval, 100, 0, selector_de_oponentes=SelectorAllForOne(agente))

    # Registro constante de entrenamiento y evaluación
    for i in range(10):
        agente.unfreeze()
        torneo_entrenar.iniciar_duelos(analisis=False, verbose=False)
        agente.freeze()
        results = torneo_probar.iniciar_duelos(analisis=True, verbose=False)
        results = float(np.mean(results[agente.__class__.__name__]))
        fila[f"mean_eval {i}"] = results
        fila[f"exploracion {i}"] = agente.porcentaje_explorado()

    return fila


def run_crecimiento_en_paralelo(configs, seeds, episodes_train=200, n_eval=10):
    tasks = []
    for cfg in configs:
        for seed in seeds:
            tasks.append((cfg, seed, episodes_train, n_eval))

    print(f"Ejecutando {len(tasks)} experimentos en paralelo...")

    # Usa todos los cores disponibles menos 1
    n_workers = max(cpu_count() - 2, 1)
    print(f"Usando {n_workers} procesos")

    rows = []
    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap_unordered(evaluar_crecimiento, tasks), total=len(tasks)):
            rows.append(result)

    return pd.DataFrame(rows)


def configs_politicas():
    politicas = ["softmax", "egreedy"]
    temperatura = [1, 2, 3]
    e_base = [0.6, 0.75, 0.9]
    e_fin = [0.3, 0.4, 0.5]
    id = 1
    configs = []

    for p in politicas:
        if p == "softmax":
            for t in temperatura:
                configs.append({
                    "tau": t, "ep_base": None, "ep_fin": None,
                    "politica": p, "id": id
                })
                id += 1
        else:
            for eb in e_base:
                for ef in e_fin:
                    configs.append({
                        "tau": None, "ep_base": eb , "ep_fin": ef,
                        "politica": p, "id": id
                    })
                    id += 1
    return configs

def crear_por_politica(config):
    politica = None
    if config["politica"] == "softmax":
        politica = Softmax(temperatura=config["tau"])
    else:
        politica = EpsilonGreedy(start_epsilon=config["ep_base"], end_epsilon=config["ep_fin"],
                              rounds_of_decay_epsilon=900000)

    agente = QLearning(politica,HistoryStatState(n_grupos=3,short_memory=15,tamaño_estado=2),alpha= 0.1, gamma=(1-(1/100)))

    return agente

def evaluar_politica(args):
    config, seed, episodes_train, n_eval = args

    random.seed(seed);
    np.random.seed(seed)

    agente = crear_por_politica(config)
    estrategias = [
        TitForTat(),
        Davis(),
        Downing(),
        Feld(),
        SiempreTraiciona(),
        agente
    ]
    fila = {**config}

    torneo_entrenar = ControladorDuelos(estrategias, episodes_train, 100, 0, selector_de_oponentes=SelectorAllForOne(agente))
    torneo_probar = ControladorDuelos(estrategias, n_eval, 100, 0, selector_de_oponentes=SelectorAllForOne(agente))

    # Registro constante de entrenamiento y evaluación
    for i in range(10):
        agente.unfreeze()
        torneo_entrenar.iniciar_duelos(analisis=False, verbose=False)
        agente.freeze()
        results = torneo_probar.iniciar_duelos(analisis=True, verbose=False)
        results = float(np.mean(results[agente.__class__.__name__]))
        fila[f"mean_eval {i}"] = results
        fila[f"exploracion {i}"] = agente.porcentaje_explorado()

    return fila

def run_crecimiento_en_paralelo_politicas(configs, seeds, episodes_train=200, n_eval=10):
    tasks = []
    for cfg in configs:
        for seed in seeds:
            tasks.append((cfg, seed, episodes_train, n_eval))

    print(f"Ejecutando {len(tasks)} experimentos en paralelo...")

    # Usa todos los cores disponibles menos 1
    n_workers = max(cpu_count() - 2, 1)
    print(f"Usando {n_workers} procesos")

    rows = []
    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap_unordered(evaluar_politica, tasks), total=len(tasks)):
            rows.append(result)

    return pd.DataFrame(rows)