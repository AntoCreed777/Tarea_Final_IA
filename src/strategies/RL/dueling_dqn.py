import math
import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle 
import os

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies

# Typing
Jugada = tuple[Elecciones, Elecciones]
Estado = tuple[Jugada, ...]
Accion = Elecciones


# -------------------------
# Estado: codificación simple
# -------------------------
def encode_estado(historial: list[Jugada], tamaño_estado: int) -> np.ndarray:
    recent = historial[-tamaño_estado:]
    vec = np.zeros((tamaño_estado * 2,), dtype=np.float32)
    offset = tamaño_estado - len(recent)
    for i, (mi, su) in enumerate(recent):
        idx = (offset + i) * 2
        vec[idx] = 0.0 if mi == Elecciones.COOPERAR else 1.0
        vec[idx + 1] = 0.0 if su == Elecciones.COOPERAR else 1.0
    return vec


# -------------------------
# Noisy Linear (Factorized Gaussian Noise, Fortunato et al.)
# -------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Mu and sigma params
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers for epsilon noise
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialization from NoisyNet paper
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        # outer product = factorized gaussian
        # use torch.outer to compute the outer product explicitly (avoids issues with .mm typing)
        outer = torch.outer(eps_out, eps_in)
        # copy_ updates the tensor contents in-place and avoids __setitem__ on Module typing
        self.weight_epsilon.copy_(outer)
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# -------------------------
# Dueling Q-network with Noisy layers
# -------------------------
class DuelingNoisyQNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        # Shared feature layer
        self.fc1 = NoisyLinear(input_dim, hidden)
        self.fc2 = NoisyLinear(hidden, hidden)

        # Value head
        self.value_fc = NoisyLinear(hidden, hidden)
        self.value_out = NoisyLinear(hidden, 1)

        # Advantage head
        self.adv_fc = NoisyLinear(hidden, hidden)
        self.adv_out = NoisyLinear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = F.relu(self.value_fc(x))
        v = self.value_out(v)  # shape (batch, 1)

        a = F.relu(self.adv_fc(x))
        a = self.adv_out(a)  # shape (batch, n_actions)

        # Combine: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        # reset noise in all NoisyLinear layers
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# -------------------------
# Replay buffer (uniform)
# -------------------------
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


# -------------------------
# Dueling Noisy DQN agent
# -------------------------
class DuelingDQN(base_strategies):
    def __init__(
        self,
        tamaño_estado: int = 5,
        gamma: float = 0.95,
        replay_capacity: int = 20000,
        batch_size: int = 128,
        device: str | None = None,
        tau: float = 0.005,
        lr: float = 5e-4,
        hidden: int = 128,
        use_opponent_context: bool = False,
        context_window: int = 10,
    ):
        """
        Dueling DQN con NoisyNet (Factorized Gaussian), Double DQN targets y soft updates.
        - tamaño_estado: número de jugadas previas codificadas.
        - use_opponent_context: añade 2 features extras (coop_rate, betray_flag).
        """
        super().__init__()

        if tamaño_estado <= 0:
            raise ValueError("tamaño_estado debe ser > 0")
        self.tamaño_estado = tamaño_estado
        self.gamma = gamma

        # replay + training
        self.replay = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size
        self.step_count = 0

        # device
        self.device = (
            torch.device(device)
            if device
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

        # features dims
        extra = 2 if use_opponent_context else 0
        input_dim = tamaño_estado * 2 + extra

        # networks
        self.policy_net = DuelingNoisyQNet(
            input_dim=input_dim, hidden=hidden, n_actions=2
        ).to(self.device)
        self.target_net = DuelingNoisyQNet(
            input_dim=input_dim, hidden=hidden, n_actions=2
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optim = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-6)

        # soft update rate
        self.tau = tau

        # internal
        self.historial: list[Jugada] = []
        self.ultimo_estado: np.ndarray | None = None
        self.ultima_accion: Accion | None = None
        self.use_opponent_context = use_opponent_context
        self.context_window = context_window

        # loss
        self.loss_fn = nn.SmoothL1Loss()
        self.actual_loss = 0.0
        self.frozen = False

    # -------------------------
    # context features (oponente)
    # -------------------------
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

    # -------------------------
    # elegir acción (sin epsilon)
    # -------------------------
    def _accion_from_policy(self, estado_vec: np.ndarray) -> Accion:
        # reset noise so different evaluations produce exploration
        self.policy_net.reset_noise()
        s_v = torch.tensor(
            estado_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(s_v)
            idx = int(torch.argmax(q, dim=1).item())
        return Elecciones.COOPERAR if idx == 0 else Elecciones.TRAICIONAR

    def realizar_eleccion(self) -> Elecciones:
        base_vec = encode_estado(self.historial, self.tamaño_estado)
        ctx_vec = self._context_features()
        estado_vec = np.concatenate([base_vec, ctx_vec]) if ctx_vec.size else base_vec

        accion = self._accion_from_policy(estado_vec)

        self.ultimo_estado = estado_vec
        self.ultima_accion = accion

        return accion

    # -------------------------
    # recibir respuesta del oponente y entrenar
    # -------------------------
    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        # si no había acción previa, registramos la jugada del oponente y salimos
        if self.ultimo_estado is None or self.ultima_accion is None:
            self.historial.append((Elecciones.COOPERAR, eleccion))
            self.ultimo_estado = None
            self.ultima_accion = None
            return

        # registrar transición
        self.historial.append((self.ultima_accion, eleccion))

        # nuevo estado
        base_vec2 = encode_estado(self.historial, self.tamaño_estado)
        ctx_vec2 = self._context_features()
        nuevo_estado_vec = (
            np.concatenate([base_vec2, ctx_vec2]) if ctx_vec2.size else base_vec2
        )

        # calcular recompensa (mantener relación original)
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

        # normalize reward to keep magnitudes reasonable (optional)
        recompensa = recompensa / 5.0  # now in [0.0, 1.0]

        action_idx = 0 if self.ultima_accion == Elecciones.COOPERAR else 1
        # Si tienes episodios finitos, marca done=True al final del duelo.
        done = False

        # push a replay buffer
        self.replay.push(
            self.ultimo_estado, action_idx, recompensa, nuevo_estado_vec, done
        )

        # train step
        self._train_step()

        # soft update target network (every step)
        with torch.no_grad():
            for tparam, param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                tparam.data.mul_(1.0 - self.tau)
                tparam.data.add_(self.tau * param.data)

        # limpiar últimos
        self.ultimo_estado = None
        self.ultima_accion = None
        self.step_count += 1

    # -------------------------
    # entrenamiento (Double DQN + Huber loss)
    # -------------------------
    def _train_step(self):
        if self.frozen or len(self.replay) < self.batch_size:
            return

        # sample
        s, a, r, s2, done = self.replay.sample(self.batch_size)
        s_v = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_v = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_v = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2_v = torch.tensor(s2, dtype=torch.float32, device=self.device)
        done_v = torch.tensor(done, dtype=torch.float32, device=self.device)

        # reset noise before computing q-values for training (policy & target)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Q(s,a) from policy net
        q_vals = self.policy_net(s_v).gather(1, a_v).squeeze(1)

        # Double DQN target:
        # next action from policy_net, value from target_net
        with torch.no_grad():
            next_actions = self.policy_net(s2_v).argmax(dim=1)
            next_q_target = (
                self.target_net(s2_v).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            )
            target = r_v + self.gamma * next_q_target * (1.0 - done_v)

        loss = self.loss_fn(q_vals, target)
        self.actual_loss = loss.item()

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optim.step()

    # -------------------------
    # reset para nuevo oponente
    # -------------------------
    def notificar_nuevo_oponente(self) -> None:
        self.historial = []
        self.ultimo_estado = None
        self.ultima_accion = None

    def get_puntaje_acumulado(self) -> str:
        return "\033[36m" + f"{super().get_puntaje_acumulado()}" + "\033[0m"

    def get_puntaje_de_este_torneo(self) -> str:
        return "\033[36m" + f"{super().get_puntaje_de_este_torneo()}" + "\033[0m"

    def get_loss(self) -> float:
        """
        Retorna el valor de pérdida (loss) del último paso de optimización.
        Returns:
            float: Valor de pérdida del último paso de optimización.
        """
        return self.actual_loss
    
    def freeze(self):
        """
        Congela el aprendizaje del agente y guarda su configuración
        de aprendizaje
        """
        self.policy_net.eval()
        self.frozen = True
        for param in self.policy_net.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Descongela el aprendizaje del agente para reanudar su configuración
        de aprendizaje
        """
        self.policy_net.train()
        self.frozen = False
        for param in self.policy_net.parameters():
            param.requires_grad = True
        
    def save(self, file: str) -> None:
        """
        Guarda el estado del agente DuelingDQN de forma segura usando `torch.save`.
        """
        os.makedirs("QTables", exist_ok=True)

        payload = {
            "config": {
                "tamaño_estado": self.tamaño_estado,
                "gamma": self.gamma,
                "replay_capacity": self.replay.buf.maxlen if hasattr(self.replay, "buf") else None,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "use_opponent_context": self.use_opponent_context,
                "context_window": self.context_window,
            },
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optim_state": self.optim.state_dict(),
            "metrics": {"actual_loss": self.actual_loss, "step_count": self.step_count},
        }

        torch.save(payload, os.path.join("QTables", f"{file}.pt"))

    @staticmethod
    def load(path: str, device: str | None = None) -> "DuelingDQN":
        payload = torch.load(path, map_location="cpu")
        cfg = payload.get("config", {})

        agent = DuelingDQN(
            tamaño_estado=cfg.get("tamaño_estado", 5),
            gamma=cfg.get("gamma", 0.95),
            replay_capacity=cfg.get("replay_capacity", 20000) or 20000,
            batch_size=cfg.get("batch_size", 128),
            tau=cfg.get("tau", 0.005),
            device=device,
            use_opponent_context=cfg.get("use_opponent_context", False),
            context_window=cfg.get("context_window", 10),
        )

        policy_state = payload.get("policy_state")
        target_state = payload.get("target_state")
        if policy_state:
            agent.policy_net.load_state_dict(policy_state)
        if target_state:
            agent.target_net.load_state_dict(target_state)

        optim_state = payload.get("optim_state")
        if optim_state:
            agent.optim.load_state_dict(optim_state)

        metrics = payload.get("metrics", {})
        agent.actual_loss = metrics.get("actual_loss", 0.0)
        agent.step_count = metrics.get("step_count", 0)

        if device:
            agent.device = (
                torch.device(device)
                if device
                else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            )
            agent.policy_net.to(agent.device)
            agent.target_net.to(agent.device)

        return agent