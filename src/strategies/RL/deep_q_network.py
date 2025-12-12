import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle 
import os

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies

# Typing
Jugada = tuple[Elecciones, Elecciones]
Estado = tuple[Jugada, ...]
Accion = Elecciones


def encode_estado(historial: list[Jugada], tamaño_estado: int) -> np.ndarray:
    """
    Codifica el historial reciente en un vector fijo de longitud `tamaño_estado * 2`.

    El vector contiene, por cada ronda en la ventana:
    - índice par: 0 si el agente cooperó, 1 si traicionó.
    - índice impar: 0 si el oponente cooperó, 1 si traicionó.

    Si el historial tiene menos de `tamaño_estado` rondas, se rellena al inicio
    con ceros para conservar una longitud constante.

    Args:
        historial (list[Jugada]): Lista de tuplas (mi_elección, su_elección) por ronda.
        tamaño_estado (int): Número de rondas recientes a codificar.

    Returns:
        np.ndarray: Vector de estado de tamaño `tamaño_estado * 2`.
    """
    recent = historial[-tamaño_estado:]
    vec = np.zeros((tamaño_estado * 2,), dtype=np.float32)
    offset = tamaño_estado - len(recent)
    for i, (mi, su) in enumerate(recent):
        idx = (offset + i) * 2
        vec[idx] = 0.0 if mi == Elecciones.COOPERAR else 1.0
        vec[idx + 1] = 0.0 if su == Elecciones.COOPERAR else 1.0
    return vec


class SimpleQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        n_actions: int = 2,
        num_hidden_layers: int = 2,
        activation: nn.Module | None = None,
    ):
        """
        Red neuronal MLP para aproximar Q(s, a).

        - input_dim: dimensión del estado (historial codificado + contexto opcional)
        - hidden: tamaño de cada capa oculta
        - n_actions: número de acciones (2: cooperar/traicionar)
        - num_hidden_layers: cantidad de capas ocultas (dinámico)
        - activation: función de activación a usar (por defecto LeakyReLU)
        """
        super().__init__()
        act = activation if activation is not None else nn.LeakyReLU()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(0, num_hidden_layers)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act)
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Devuelve los valores Q para cada acción dado el estado x (tensor 2D)."""
        return self.net(x)


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        Memoria de repetición (FIFO con tamaño fijo) para almacenar transiciones.

        Args:
            capacity (int): Número máximo de transiciones a conservar.
        """
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        """
        Inserta una transición en el buffer.

        Args:
            s (np.ndarray): Estado actual codificado.
            a (int): Índice de acción ejecutada (0: cooperar, 1: traicionar).
            r (float): Recompensa recibida.
            s2 (np.ndarray): Estado siguiente codificado.
            done (bool): Indicador de finalización del episodio/duelo.
        """
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        """
        Devuelve un lote aleatorio de transiciones para entrenamiento.

        Args:
            batch_size (int): Número de transiciones a muestrear.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Batch con (s, a, r, s2, done) apilados y tipados.
        """
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.uint8),
        )

    def __len__(self):
        """Returns: int: Cantidad de transiciones almacenadas actualmente."""
        return len(self.buf)


class DeepQNetwork(base_strategies):
    def __init__(
        self,
        tamaño_estado: int = 5,
        alpha: float = 1e-3,
        gamma: float = 0.95,
        start_epsilon: float = 0.5,
        end_epsilon: float = 0.05,
        rounds_of_decay_epsilon: int = 1000,
        replay_capacity: int = 2000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        device: str | None = None,
        log_interval: int = 500,
        use_opponent_context: bool = False,
        context_window: int = 10,
        num_hidden_layers: int = 2,
    ):
        """
        Implementación de una estrategia basada en Deep Q-Network para el
        Dilema del Prisionero iterado.

        La red de política (policy_net) estima Q(s, a) y se actualiza mediante
        muestras del buffer de repetición. La red objetivo (target_net) se
        sincroniza suavemente en cada paso para estabilizar el aprendizaje.

        Args:
            tamaño_estado (int, default=5): Longitud de la ventana del historial codificado.
            alpha (float, default=1e-3): Tasa de aprendizaje del optimizador Adam.
            gamma (float, default=0.95): Factor de descuento para recompensas futuras.
            start_epsilon (float, default=0.5): Exploración inicial (epsilon-greedy).
            end_epsilon (float, default=0.05): Exploración mínima.
            rounds_of_decay_epsilon (int, default=1000): Rondas para el decaimiento lineal de epsilon.
            replay_capacity (int, default=2000): Capacidad máxima del buffer de repetición.
            batch_size (int, default=64): Tamaño del lote de entrenamiento.
            target_update_freq (int, default=200): Frecuencia de actualización dura de la red objetivo (no usada si se aplica soft-update por `tau`).
            device (str | None, default=None): Dispositivo ('cuda'/'cpu'); autodetección por defecto.
            log_interval (int, default=500): Intervalo de pasos para registros (si se habilitan).
            use_opponent_context (bool, default=False): Si añade contexto del oponente al estado.
            context_window (int, default=10): Ventana para calcular el contexto del oponente.
            num_hidden_layers (int, default=2): Número de capas ocultas en la MLP.

        Atributos heredados:
            puntaje (int): Acumulador de recompensas definido en la clase base.

        Atributos:
            policy_net (nn.Module): Red que estima Q(s, a) y se entrena.
            target_net (nn.Module): Copia de `policy_net` para objetivos estables.
            optim (optim.Optimizer): Optimizador Adam para la red de política.
            replay (ReplayBuffer): Memoria de repetición para transiciones.
            historial (list[Jugada]): Historial de jugadas en el duelo actual.
            ultimo_estado (np.ndarray | None): Último estado codificado observado.
            ultima_accion (Accion | None): Última acción propia ejecutada.
            epsilon (float): Nivel actual de exploración.
            end_epsilon (float): Mínimo de exploración.
            epsilon_decay (float): Paso de decaimiento lineal por ronda.
            tau (float): Tasa de actualización suave para la red objetivo.
            step_count (int): Contador de pasos de entrenamiento/progreso.
            device (torch.device): Dispositivo de ejecución.

        Notas:
            - Se normaliza la recompensa a [0, 1] dividiendo por 5.0.
            - Epsilon decrece linealmente en cada `realizar_eleccion()`.
            - Double DQN: la acción siguiente se elige con `policy_net` y su valor
              se evalúa con `target_net`.
            - Si `device` no se especifica y hay CUDA disponible, se usa GPU.
        """
        super().__init__()
        if tamaño_estado <= 0:
            raise ValueError("tamaño_estado debe ser > 0")
        self.tamaño_estado = tamaño_estado
        self.alpha = alpha
        self.gamma = gamma
        self.tau = 0.005

        # Plan de exploración epsilon-greedy (decaimiento lineal)
        if start_epsilon < end_epsilon:
            raise ValueError("start_epsilon >= end_epsilon")
        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = (start_epsilon - end_epsilon) / max(
            1, rounds_of_decay_epsilon
        )

        # Memoria de repetición + parámetros de entrenamiento
        self.replay = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.log_interval = log_interval

        # CPU o GPU para tensores
        self.device = (
            torch.device(device)
            if device
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

        # Redes: policy y target comparten arquitectura; target se sincroniza con soft update
        input_dim = tamaño_estado * 2 + (2 if use_opponent_context else 0)
        self.num_hidden_layers = num_hidden_layers
        self.policy_net = SimpleQNet(
            input_dim, hidden=64, n_actions=2, num_hidden_layers=self.num_hidden_layers
        ).to(self.device)
        self.target_net = SimpleQNet(
            input_dim, hidden=64, n_actions=2, num_hidden_layers=self.num_hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optim = optim.Adam(
            self.policy_net.parameters(), lr=alpha, weight_decay=1e-5
        )

        # Estado interno del agente (historial y último estado/acción observados)
        self.historial: list[Jugada] = []
        self.ultimo_estado: np.ndarray | None = None
        self.ultima_accion: Accion | None = None
        self.use_opponent_context = use_opponent_context
        self.context_window = context_window
        self.actual_loss = 0.0
        self.frozen = False

    def _context_features(self) -> np.ndarray:
        """
        Devuelve características agregadas del comportamiento reciente del oponente:
        - tasa de cooperación en una ventana
        - flag binario si hubo alguna traición reciente
        """
        if not self.use_opponent_context:
            return np.zeros((0,), dtype=np.float32)
        window = self.historial[-self.context_window :]
        if not window:
            return np.array([0.0, 0.0], dtype=np.float32)
        coop_count = sum(1 for _, su in window if su == Elecciones.COOPERAR)
        betray_any = any(su == Elecciones.TRAICIONAR for _, su in window)
        coop_rate = coop_count / len(window)
        betray_flag = 1.0 if betray_any else 0.0
        """
        Returns:
            np.ndarray: Vector `[coop_rate, betray_flag]` de tipo float32.
        """
        return np.array([coop_rate, betray_flag], dtype=np.float32)

    def _accion_eps_greedy(self, estado_vec: np.ndarray) -> Accion:
        """
        Selecciona una acción mediante una política epsilon-greedy.

        Con probabilidad `epsilon`, elige aleatoriamente entre cooperar o
        traicionar. En caso contrario, selecciona la acción con mayor Q(s, a)
        según la red de política.

        Args:
            estado_vec (np.ndarray): Estado codificado actual.

        Returns:
            Accion: Acción seleccionada (Elecciones.COOPERAR o Elecciones.TRAICIONAR).
        """
        if random.random() < self.epsilon:
            return random.choice([Elecciones.COOPERAR, Elecciones.TRAICIONAR])
        s_v = torch.tensor(
            estado_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(s_v)
            idx = int(torch.argmax(q, dim=1).item())
        return Elecciones.COOPERAR if idx == 0 else Elecciones.TRAICIONAR

    def realizar_eleccion(self) -> Elecciones:
        """
        Construye el estado actual (historial codificado + contexto opcional),
        selecciona la acción mediante `_accion_eps_greedy` y actualiza `epsilon`.

        Returns:
            Elecciones: Acción elegida para la ronda actual.
        """
        base_vec = encode_estado(self.historial, self.tamaño_estado)
        ctx_vec = self._context_features()
        estado_vec = np.concatenate([base_vec, ctx_vec]) if ctx_vec.size else base_vec
        accion = self._accion_eps_greedy(estado_vec)

        self.ultimo_estado = estado_vec
        self.ultima_accion = accion

        # decrecer epsilon (lineal)
        self.epsilon = max(self.end_epsilon, self.epsilon - self.epsilon_decay)

        return accion

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Recibe la acción del oponente, registra la transición correspondiente y
        ejecuta un paso de entrenamiento si hay suficientes muestras.

        En la primera ronda, puede ser llamada antes de contar con
        `ultima_accion`/`ultimo_estado`; no se fabrica una acción previa.

        Args:
            eleccion (Elecciones): Acción realizada por el oponente en esta ronda.
        """
        if self.ultimo_estado is None or self.ultima_accion is None:
            if self.ultima_accion is not None:
                self.historial.append((self.ultima_accion, eleccion))
            self.ultimo_estado = None
            self.ultima_accion = None
            return

        # registrar transición
        mi_idx = 0 if self.ultima_accion == Elecciones.COOPERAR else 1
        su_idx = 0 if eleccion == Elecciones.COOPERAR else 1
        self.historial.append((self.ultima_accion, eleccion))

        base_vec2 = encode_estado(self.historial, self.tamaño_estado)
        ctx_vec2 = self._context_features()
        nuevo_estado_vec = (
            np.concatenate([base_vec2, ctx_vec2]) if ctx_vec2.size else base_vec2
        )

        # calcular recompensa
        if (
            self.ultima_accion == Elecciones.COOPERAR
            and eleccion == Elecciones.COOPERAR
        ):
            recompensa = 3.0
        elif (
            self.ultima_accion == Elecciones.COOPERAR
            and eleccion == Elecciones.TRAICIONAR
        ):
            recompensa = 0.0
        elif (
            self.ultima_accion == Elecciones.TRAICIONAR
            and eleccion == Elecciones.COOPERAR
        ):
            recompensa = 5.0
        else:
            recompensa = 1.0

        # Normalización de la recompensa a [0,1]
        recompensa = recompensa / 5.0

        action_idx = mi_idx
        done = False

        # Guardar transición en el replay buffer
        self.replay.push(
            self.ultimo_estado, action_idx, recompensa, nuevo_estado_vec, done
        )

        # Entrenar un paso (si hay suficientes muestras)
        self._train_step()

        # actualizar contadores / limpiar ultimo state para próximo paso
        self.ultimo_estado = None
        self.ultima_accion = None
        self.step_count += 1
        # Soft update de la red objetivo para mayor estabilidad
        for target_param, param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * param.data)

    def _train_step(self):
        """
        Ejecuta un paso de entrenamiento de Double DQN a partir de transiciones
        muestreadas del buffer.

        Proceso:
        1. Se calcula Q(s, a) actual con la red de política.
        2. Se selecciona la acción siguiente con la red de política y se evalúa
           su valor con la red objetivo (Double DQN).
        3. Se construye el objetivo y se optimiza la red de política mediante
           pérdida Huber (SmoothL1Loss).
        """
        if self.frozen or len(self.replay) < self.batch_size:
            return
        s, a, r, s2, done = self.replay.sample(self.batch_size)

        s_v = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_v = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_v = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2_v = torch.tensor(s2, dtype=torch.float32, device=self.device)
        done_v = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Q(s,a) actuales de la red de política
        q_vals = self.policy_net(s_v).gather(1, a_v).squeeze(1)
        with torch.no_grad():
            # Double DQN: acción siguiente con la policy, valor con la target
            next_actions = self.policy_net(s2_v).argmax(dim=1)
            next_q_target = (
                self.target_net(s2_v).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            )
            target = r_v + self.gamma * next_q_target * (1.0 - done_v)

        loss = nn.SmoothL1Loss()(q_vals, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optim.step()
        self.actual_loss = loss.item()

    def get_loss(self) -> float:
        """
        Retorna el historial de pérdidas (Huber/TD error) acumuladas por paso
        de entrenamiento para poder graficar y evaluar convergencia.

        Returns:
            list[float]: Lista de valores de pérdida por cada _train_step.
        """
        return self.actual_loss

    def notificar_nuevo_oponente(self) -> None:
        """
        Resetea el estado interno para comenzar la interacción contra un
        oponente completamente nuevo, reiniciando las observaciones acumuladas
        del duelo en curso.
        """
        self.historial = []
        self.ultimo_estado = None
        self.ultima_accion = None

    def get_puntaje_acumulado(self) -> str:
        return "\033[34m" + f"{super().get_puntaje_acumulado()}" + "\033[0m"

    def get_puntaje_de_este_torneo(self) -> str:
        return "\033[34m" + f"{super().get_puntaje_de_este_torneo()}" + "\033[0m"

    def freeze(self):
        """Congela los parámetros de la red y deshabilita entrenamiento."""
        self.policy_net.eval()
        self.frozen = True
        for param in self.policy_net.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Descongela los parámetros de la red y habilita entrenamiento."""
        self.policy_net.train()
        self.frozen = False
        for param in self.policy_net.parameters():
            param.requires_grad = True

    def save(self, file: str) -> None:
        """
        Guarda el estado entrenable del agente de forma segura.

        Se evitan objetos no serializables (como el propio módulo/instancia) y se
        almacena un diccionario con:
        - parámetros de configuración
        - `policy_net.state_dict()` y `target_net.state_dict()`
        - `optim.state_dict()`
        - métricas ligeras (`actual_loss`, `step_count`)

        Args:
            file (str): Nombre base del archivo a guardar (sin extensión).
        """
        os.makedirs("QTables", exist_ok=True)

        payload = {
            "config": {
                "tamaño_estado": self.tamaño_estado,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "start_epsilon": self.epsilon,  # epsilon actual
                "end_epsilon": self.end_epsilon,
                "epsilon_decay": self.epsilon_decay,
                "replay_capacity": self.replay.buf.maxlen if hasattr(self.replay.buf, "maxlen") else None,
                "batch_size": self.batch_size,
                "target_update_freq": self.target_update_freq,
                "use_opponent_context": self.use_opponent_context,
                "context_window": self.context_window,
                "num_hidden_layers": self.num_hidden_layers,
                "tau": self.tau,
            },
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optim_state": self.optim.state_dict(),
            "metrics": {
                "actual_loss": self.actual_loss,
                "step_count": self.step_count,
            },
        }

        torch.save(payload, os.path.join("QTables", f"{file}.pt"))

    @staticmethod
    def load(path: str, device: str | None = None) -> "DeepQNetwork":
        """
        Carga un agente desde un archivo guardado por `save`.

        Args:
            path (str): Ruta al archivo `.pt` guardado.
            device (str | None): Dispositivo deseado ('cuda'/'cpu'). Si es None,
                                 se usa autodetección como en el constructor.

        Returns:
            DeepQNetwork: Instancia reconstruida con pesos y optimizador restaurados.
        """
        payload = torch.load(path, map_location="cpu")

        cfg = payload.get("config", {})
        # Reconstruir con configuración mínima necesaria
        agent = DeepQNetwork(
            tamaño_estado=cfg.get("tamaño_estado", 5),
            alpha=cfg.get("alpha", 1e-3),
            gamma=cfg.get("gamma", 0.95),
            start_epsilon=cfg.get("start_epsilon", 0.5),
            end_epsilon=cfg.get("end_epsilon", 0.05),
            rounds_of_decay_epsilon=int((cfg.get("start_epsilon", 0.5) - cfg.get("end_epsilon", 0.05)) / max(1e-12, cfg.get("epsilon_decay", 1e-3))),
            replay_capacity=cfg.get("replay_capacity", 2000) or 2000,
            batch_size=cfg.get("batch_size", 64),
            target_update_freq=cfg.get("target_update_freq", 200),
            device=device,
            use_opponent_context=cfg.get("use_opponent_context", False),
            context_window=cfg.get("context_window", 10),
            num_hidden_layers=cfg.get("num_hidden_layers", 2),
        )

        # Cargar pesos de redes
        policy_state = payload.get("policy_state")
        target_state = payload.get("target_state")
        if policy_state:
            agent.policy_net.load_state_dict(policy_state)
        if target_state:
            agent.target_net.load_state_dict(target_state)

        # Cargar estado del optimizador
        optim_state = payload.get("optim_state")
        if optim_state:
            agent.optim.load_state_dict(optim_state)

        # Métricas
        metrics = payload.get("metrics", {})
        agent.actual_loss = metrics.get("actual_loss", 0.0)
        agent.step_count = metrics.get("step_count", 0)

        # Restaurar epsilon actual si viene en config
        if "start_epsilon" in cfg:
            agent.epsilon = cfg.get("start_epsilon", agent.epsilon)

        # Ajustar device si se solicita diferente
        if device:
            agent.device = (
                torch.device(device)
                if device
                else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            )
            agent.policy_net.to(agent.device)
            agent.target_net.to(agent.device)

        return agent