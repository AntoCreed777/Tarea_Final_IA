# a2c_strategy.py
import random
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies

def encode_estado(historial: list[tuple[Elecciones, Elecciones]], tamaño_estado: int) -> np.ndarray:
    recent = historial[-tamaño_estado:]
    vec = np.zeros((tamaño_estado * 2,), dtype=np.float32)
    offset = tamaño_estado - len(recent)
    for i, (mi, su) in enumerate(recent):
        idx = (offset + i) * 2
        vec[idx] = 0.0 if mi == Elecciones.COOPERAR else 1.0
        vec[idx + 1] = 0.0 if su == Elecciones.COOPERAR else 1.0
    return vec

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value

class A2C(base_strategies):
    def __init__(
        self,
        tamaño_estado: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.95,
        entropy_coef: float = 1e-3,
        value_coef: float = 0.5,
        device: str = None,
    ):
        super().__init__()
        if tamaño_estado <= 0:
            raise ValueError("tamaño_estado > 0 required")
        self.tamaño_estado = tamaño_estado
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        input_dim = tamaño_estado * 2
        self.net = ActorCriticNet(input_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.historial: list[tuple[Elecciones, Elecciones]] = []
        self.ultimo_estado_vec = None
        self.ultima_accion = None
        self.ultimo_valor = None

    def realizar_eleccion(self) -> Elecciones:
        estado_vec = encode_estado(self.historial, self.tamaño_estado)
        s_v = torch.tensor(estado_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(s_v)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        act = dist.sample().item()
        self.ultima_accion = Elecciones.COOPERAR if act == 0 else Elecciones.TRAICIONAR
        self.ultimo_estado_vec = estado_vec
        self.ultimo_valor = value.detach().squeeze(0).item()
        return self.ultima_accion

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        if self.ultima_accion is None or self.ultimo_estado_vec is None:
            self.historial.append((self.ultima_accion if self.ultima_accion else Elecciones.COOPERAR, eleccion))
            self.ultimo_accion = None
            self.ultimo_estado_vec = None
            self.ultimo_valor = None
            return

        if self.ultima_accion == Elecciones.COOPERAR and eleccion == Elecciones.COOPERAR:
            recompensa = 3.0
        elif self.ultima_accion == Elecciones.COOPERAR and eleccion == Elecciones.TRAICIONAR:
            recompensa = 0.0
        elif self.ultima_accion == Elecciones.TRAICIONAR and eleccion == Elecciones.COOPERAR:
            recompensa = 5.0
        else:
            recompensa = 1.0

        # registrar jugada
        self.historial.append((self.ultima_accion, eleccion))

        # bootstrap: obtener valor del nuevo estado
        nuevo_estado_vec = encode_estado(self.historial, self.tamaño_estado)
        s2_v = torch.tensor(nuevo_estado_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.net(s2_v)
            next_value = next_value.squeeze(0).item()

        # calcular TD error
        td_target = recompensa + self.gamma * next_value
        advantage = td_target - (self.ultimo_valor if self.ultimo_valor is not None else 0.0)

        # preparar tensors para update
        s_v = torch.tensor(self.ultimo_estado_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value_pred = self.net(s_v)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_idx = 0 if self.ultima_accion == Elecciones.COOPERAR else 1
        action_logprob = log_probs[0, action_idx]
        entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum()

        # pérdidas
        policy_loss = -action_logprob * advantage
        value_loss = nn.MSELoss()(value_pred.squeeze(0), torch.tensor(td_target, dtype=torch.float32, device=self.device))
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # paso de optimización
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()

        # limpiar últimos
        self.ultimo_estado_vec = None
        self.ultima_accion = None
        self.ultimo_valor = None

    def notificar_nuevo_oponente(self) -> None:
        self.historial = []
        self.ultimo_estado_vec = None
        self.ultima_accion = None
        self.ultimo_valor = None

    def get_puntaje_acumulado(self) -> str:
        return "\033[35m" + f"{super().get_puntaje_acumulado()}" + "\033[0m"

    def get_puntaje_de_este_torneo(self) -> str:
        return "\033[35m" + f"{super().get_puntaje_de_este_torneo()}" + "\033[0m"
