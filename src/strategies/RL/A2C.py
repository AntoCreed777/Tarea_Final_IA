import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle 
import os

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


def encode_estado(
    historial: list[tuple[Elecciones, Elecciones]], tamaño_estado: int
) -> np.ndarray:
    """
    Codifica el historial reciente en un vector fijo de longitud `tamaño_estado * 2`.

    El vector contiene, por cada ronda en la ventana:
    - índice par: 0 si el agente cooperó, 1 si traicionó.
    - índice impar: 0 si el oponente cooperó, 1 si traicionó.

    Si el historial tiene menos de `tamaño_estado` rondas, se rellena al inicio
    con ceros para conservar una longitud constante.

    Args:
        historial (list[tuple[Elecciones, Elecciones]]): Lista de tuplas
            (mi_elección, su_elección) por ronda.
        tamaño_estado (int): Número de rondas recientes a codificar.

    Returns:
        np.ndarray: Vector de estado de tamaño `tamaño_estado * 2` y dtype float32.
    """
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
        """
        Red neuronal Actor-Crítico con bloque compartido y dos cabezas.

        Arquitectura:
        - Bloque compartido: Linear(input_dim → hidden) + LeakyReLU
        - Política (actor): Linear(hidden → hidden) + LeakyReLU + Linear(hidden → 2)
        - Valor (crítico): Linear(hidden → hidden) + LeakyReLU + Linear(hidden → 1)

        Args:
            input_dim (int): Dimensión del estado de entrada (historial codificado).
            hidden (int, default=64): Tamaño de la capa oculta compartida y de cabezas.

        Atributos:
            shared (nn.Sequential): Bloque compartido para extracción de características.
            policy (nn.Sequential): Cabeza que produce logits de acción.
            value (nn.Sequential): Cabeza que estima V(s).
        """
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden), nn.LeakyReLU())
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, 2)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        """
        FeedForward.

        Args:
            x (torch.Tensor): Tensor 2D con forma `[batch, input_dim]`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - logits (torch.Tensor): Logits de la política con forma `[batch, 2]`.
                - value (torch.Tensor): Valor estimado V(s) con forma `[batch]`.
        """
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
        device: Optional[str] = None,
    ):
        """
        Estrategia Actor-Crítico (A2C) para el Dilema del Prisionero iterado.

        El actor maximiza la probabilidad de acciones con ventaja positiva
        (policy gradient), mientras el crítico estima V(s) y reduce el error
        de valor (MSE). Se añade una entropía para promover exploración.

        Args:
            tamaño_estado (int, default=5): Longitud de la ventana del historial codificado.
            lr (float, default=1e-3): Tasa de aprendizaje del optimizador Adam.
            gamma (float, default=0.95): Factor de descuento para TD target.
            entropy_coef (float, default=1e-3): Coeficiente de entropía en la pérdida.
            value_coef (float, default=0.5): Peso de la pérdida del crítico (MSE).
            device (str | None, default=None): Dispositivo ('cuda'/'cpu'); autodetección por defecto.

        Atributos heredados:
            puntaje (int): Acumulador de recompensas definido en la clase base.

        Atributos:
            net (ActorCriticNet): Red actor-crítico.
            opt (optim.Optimizer): Optimizador Adam.
            historial (list[tuple[Elecciones, Elecciones]]): Historial de jugadas.
            ultimo_estado_vec (np.ndarray | None): Último estado codificado observado.
            ultima_accion (Elecciones | None): Última acción propia ejecutada.
            ultimo_valor (float | None): V(s) del último estado observado.

        Notas:
            - La entropía regulariza la política para evitar colapso temprano.
            - Se realiza una actualización por paso tras observar la acción del oponente.
        """
        super().__init__()
        if tamaño_estado <= 0:
            raise ValueError("tamaño_estado > 0 required")
        self.tamaño_estado = tamaño_estado
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.device = (
            torch.device(device)
            if device
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

        input_dim = tamaño_estado * 2
        self.net = ActorCriticNet(input_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.historial: list[tuple[Elecciones, Elecciones]] = []
        self.ultimo_estado_vec = None
        self.ultima_accion = None
        self.ultimo_valor = None
        self.actual_loss = 0.0

    def realizar_eleccion(self) -> Elecciones:
        """
        Construye el estado actual, muestrea una acción de la política
        categórica y guarda los valores necesarios para el paso de actualización.

        Returns:
            Elecciones: Acción elegida para la ronda actual.
        """
        estado_vec = encode_estado(self.historial, self.tamaño_estado)
        s_v = torch.tensor(
            estado_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        logits, value = self.net(s_v)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        act = dist.sample().item()
        self.ultima_accion = Elecciones.COOPERAR if act == 0 else Elecciones.TRAICIONAR
        self.ultimo_estado_vec = estado_vec
        self.ultimo_valor = value.detach().squeeze(0).item()
        return self.ultima_accion

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Recibe la acción del oponente, calcula el objetivo TD, la ventaja y
        realiza un paso de optimización para actor y crítico.

        En la primera ronda puede no existir `ultima_accion`/`ultimo_estado_vec`;
        en tal caso, se registra la jugada y se pospone el entrenamiento.

        Args:
            eleccion (Elecciones): Acción realizada por el oponente.
        """
        if self.ultima_accion is None or self.ultimo_estado_vec is None:
            self.historial.append(
                (
                    self.ultima_accion if self.ultima_accion else Elecciones.COOPERAR,
                    eleccion,
                )
            )
            self.ultimo_accion = None
            self.ultimo_estado_vec = None
            self.ultimo_valor = None
            return

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

        # registrar jugada
        self.historial.append((self.ultima_accion, eleccion))

        # bootstrap: obtener valor del nuevo estado
        nuevo_estado_vec = encode_estado(self.historial, self.tamaño_estado)
        s2_v = torch.tensor(
            nuevo_estado_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.net(s2_v)
            next_value = next_value.squeeze(0).item()

        # calcular TD error
        td_target = recompensa + self.gamma * next_value
        advantage = td_target - (
            self.ultimo_valor if self.ultimo_valor is not None else 0.0
        )

        # preparar tensors para update
        s_v = torch.tensor(
            self.ultimo_estado_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        logits, value_pred = self.net(s_v)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_idx = 0 if self.ultima_accion == Elecciones.COOPERAR else 1
        action_logprob = log_probs[0, action_idx]
        entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum()

        # pérdidas
        policy_loss = -action_logprob * advantage
        value_loss = nn.MSELoss()(
            value_pred.squeeze(0),
            torch.tensor(td_target, dtype=torch.float32, device=self.device),
        )
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # paso de optimización
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()
        self.actual_loss = loss.item()

        # limpiar últimos
        self.ultimo_estado_vec = None
        self.ultima_accion = None
        self.ultimo_valor = None

    def notificar_nuevo_oponente(self) -> None:
        """
        Resetea el estado interno para iniciar un duelo contra un oponente nuevo.
        """
        self.historial = []
        self.ultimo_estado_vec = None
        self.ultima_accion = None
        self.ultimo_valor = None

    def get_puntaje_acumulado(self) -> str:
        return "\033[35m" + f"{super().get_puntaje_acumulado()}" + "\033[0m"

    def get_puntaje_de_este_torneo(self) -> str:
        return "\033[35m" + f"{super().get_puntaje_de_este_torneo()}" + "\033[0m"

    def get_loss(self) -> float:
        """
        Retorna el valor de pérdida (loss) del último paso de optimización.
        Returns:
            float: Valor de pérdida del último paso de optimización.
        """
        return self.actual_loss
    
    def freeze(self):
        """Congela los parámetros de la red para evitar más entrenamientos."""
        for param in self.net.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Descongela los parámetros de la red para permitir entrenamientos."""
        for param in self.net.parameters():
            param.requires_grad = True

    def save(self, file : str) -> None:
        """
        Exporta la QTable para futuros agentes
        """
        # Crear carpeta si no existe
        os.makedirs("QTables", exist_ok=True)

        with open(f"Qtables/{file}.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod 
    def load(path):
        with open(path, 'rb') as f: 
            return pickle.load(f)