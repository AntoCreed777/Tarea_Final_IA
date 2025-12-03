import random
from collections import deque
from typing import Deque, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies

# Typing
Jugada = tuple[Elecciones, Elecciones]
Estado = tuple[Jugada, ...]
Accion = Elecciones

def encode_estado(historial: list[Jugada], tamaño_estado: int) -> np.ndarray:
    recent = historial[-tamaño_estado:]
    vec = np.zeros((tamaño_estado * 2,), dtype=np.float32)
    offset = tamaño_estado - len(recent)
    for i, (mi, su) in enumerate(recent):
        idx = (offset + i) * 2
        vec[idx] = 0.0 if mi == Elecciones.COOPERAR else 1.0
        vec[idx + 1] = 0.0 if su == Elecciones.COOPERAR else 1.0
    return vec

class SimpleQNet(nn.Module):
    def __init__(self, input_dim: int, hidden=64, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
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
        debug: bool = False,
        log_interval: int = 500,
        use_opponent_context: bool = False,
        context_window: int = 10,
    ):
        super().__init__()
        if tamaño_estado <= 0:
            raise ValueError("tamaño_estado debe ser > 0")
        self.tamaño_estado = tamaño_estado
        self.alpha = alpha
        self.gamma = gamma

        # epsilon schedule (lineal)
        if start_epsilon < end_epsilon:
            raise ValueError("start_epsilon >= end_epsilon")
        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = (start_epsilon - end_epsilon) / max(1, rounds_of_decay_epsilon)

        # replay + training
        self.replay = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.debug = debug
        self.log_interval = log_interval

        # device
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f"[DQN] Using device: {self.device}")

        # networks
        input_dim = tamaño_estado * 2 + (2 if use_opponent_context else 0)
        self.policy_net = SimpleQNet(input_dim).to(self.device)
        self.target_net = SimpleQNet(input_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optim = optim.Adam(self.policy_net.parameters(), lr=alpha, weight_decay=1e-5)

        # estado interno
        self.historial: list[Jugada] = []
        self.ultimo_estado: np.ndarray | None = None
        self.ultima_accion: Accion | None = None
        self.use_opponent_context = use_opponent_context
        self.context_window = context_window

    def _context_features(self) -> np.ndarray:
        if not self.use_opponent_context:
            return np.zeros((0,), dtype=np.float32)
        window = self.historial[-self.context_window :]
        if not window:
            return np.array([0.0, 0.0], dtype=np.float32)
        coop_count = sum(1 for _, su in window if su == Elecciones.COOPERAR)
        betray_any = any(su == Elecciones.TRAICIONAR for _, su in window)
        coop_rate = coop_count / len(window)
        betray_flag = 1.0 if betray_any else 0.0
        return np.array([coop_rate, betray_flag], dtype=np.float32)

    def _accion_eps_greedy(self, estado_vec: np.ndarray) -> Accion:
        if random.random() < self.epsilon:
            return random.choice([Elecciones.COOPERAR, Elecciones.TRAICIONAR])
        s_v = torch.tensor(estado_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(s_v)
            idx = int(torch.argmax(q, dim=1).item())
        if self.debug and (self.step_count % self.log_interval == 0):
            q_vals = q.squeeze(0).detach().cpu().numpy()
            print(f"[DQN] step={self.step_count} eps={self.epsilon:.3f} q={q_vals}")
        return Elecciones.COOPERAR if idx == 0 else Elecciones.TRAICIONAR

    def realizar_eleccion(self) -> Elecciones:
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
        if self.ultimo_estado is None or self.ultima_accion is None:
            # si no hay acción previa (primer paso), solo registrar
            self.historial.append((self.ultima_accion if self.ultima_accion else Elecciones.COOPERAR, eleccion))
            # Reiniciar ultimo estado para siguiente paso
            self.ultimo_estado = None
            self.ultima_accion = None
            return

        # registrar transición
        mi_idx = 0 if self.ultima_accion == Elecciones.COOPERAR else 1
        su_idx = 0 if eleccion == Elecciones.COOPERAR else 1
        self.historial.append((self.ultima_accion, eleccion))

        base_vec2 = encode_estado(self.historial, self.tamaño_estado)
        ctx_vec2 = self._context_features()
        nuevo_estado_vec = np.concatenate([base_vec2, ctx_vec2]) if ctx_vec2.size else base_vec2

        # calcular recompensa
        if self.ultima_accion == Elecciones.COOPERAR and eleccion == Elecciones.COOPERAR:
            recompensa = 3.0
        elif self.ultima_accion == Elecciones.COOPERAR and eleccion == Elecciones.TRAICIONAR:
            recompensa = 0.0
        elif self.ultima_accion == Elecciones.TRAICIONAR and eleccion == Elecciones.COOPERAR:
            recompensa = 5.0
        else:
            recompensa = 1.0

        recompensa = np.clip(recompensa, -1.0, 1.0)

        action_idx = mi_idx
        done = False

        # push a replay
        self.replay.push(self.ultimo_estado, action_idx, recompensa, nuevo_estado_vec, done)

        # entrenar paso (si hay suficientes muestras)
        self._train_step()

        self.tau = 0.005

        # actualizar contadores / limpiar ultimo state para próximo paso
        self.ultimo_estado = None
        self.ultima_accion = None
        self.step_count += 1
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * param.data)

    def _train_step(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, s2, done = self.replay.sample(self.batch_size)

        s_v = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_v = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_v = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2_v = torch.tensor(s2, dtype=torch.float32, device=self.device)
        done_v = torch.tensor(done, dtype=torch.float32, device=self.device)

        q_vals = self.policy_net(s_v).gather(1, a_v).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(s2_v).argmax(dim=1)
            next_q_target = self.target_net(s2_v).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = r_v + self.gamma * next_q_target * (1.0 - done_v)

        loss = nn.SmoothL1Loss()(q_vals, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optim.step()
        if self.debug and (self.step_count % self.log_interval == 0):
            avg_q = float(q_vals.mean().item())
            avg_t = float(target.mean().item())
            print(f"[DQN] loss={loss.item():.4f} avg_q={avg_q:.3f} avg_target={avg_t:.3f} replay={len(self.replay)}")

    def notificar_nuevo_oponente(self) -> None:
        self.historial = []
        self.ultimo_estado = None
        self.ultima_accion = None

    def get_puntaje_acumulado(self) -> str:
        return "\033[34m" + f"{super().get_puntaje_acumulado()}" + "\033[0m"
    def get_puntaje_de_este_torneo(self) -> str:
        return "\033[34m" + f"{super().get_puntaje_de_este_torneo()}" + "\033[0m"
