import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle 
import os

from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


def encode_estado(historial, tamaño_estado):
    recent = historial[-tamaño_estado:]
    vec = np.zeros((tamaño_estado * 2,), dtype=np.float32)
    offset = tamaño_estado - len(recent)
    for i, (mi, su) in enumerate(recent):
        idx = (offset + i) * 2
        vec[idx] = 0.0 if mi == Elecciones.COOPERAR else 1.0
        vec[idx + 1] = 0.0 if su == Elecciones.COOPERAR else 1.0
    return vec


# ============================================================
#       RED A2C CON LSTM COMPARTIDA
# ============================================================
class ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()

        self.hidden_size = hidden

        # LSTM compartida para aprender dependencias temporales
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)

        # Cabeza de política
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )

        # Cabeza de valor
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, hc):
        """
        x: (batch=1, seq_len=1, input_dim)
        hc: (h, c)
        """
        out, hc = self.lstm(x, hc)
        out = out[:, -1]  # última salida

        logits = self.policy(out)
        value = self.value(out).squeeze(-1)
        return logits, value, hc

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)
        return (h, c)


# ============================================================
#                ESTRATEGIA A2C + LSTM
# ============================================================
class A2C_LSTM(base_strategies):
    def __init__(
        self,
        tamaño_estado: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.95,
        entropy_coef: float = 1e-3,
        value_coef: float = 0.5,
        device: str | None = None,
        reset_memory_on_new_opponent: bool = True,
        reset_history_on_new_opponent: bool = True,
    ):
        super().__init__()
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
        self.net = ActorCriticLSTM(input_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # historial
        self.historial = []

        # memoria interna de la LSTM
        self.hc = self.net.init_hidden(batch_size=1)
        self.hc = (self.hc[0].to(self.device), self.hc[1].to(self.device))

        # buffers temporales
        self.last_state_vec = None
        self.last_action = None
        self.last_value = None
        self.last_hc = None

        # opciones de reseteo
        self.reset_memory_on_new_opponent = reset_memory_on_new_opponent
        self.reset_history_on_new_opponent = reset_history_on_new_opponent

        self.actual_loss = 0.0
        self.frozen = False

    # --------------------------------------------------------
    #                   ELEGIR ACCIÓN
    # --------------------------------------------------------
    def realizar_eleccion(self) -> Elecciones:
        estado_vec = encode_estado(self.historial, self.tamaño_estado)
        x = torch.tensor(estado_vec, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,input_dim)

        logits, value, new_hc = self.net(x, self.hc)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        self.last_state_vec = estado_vec
        self.last_action = Elecciones.COOPERAR if action == 0 else Elecciones.TRAICIONAR
        self.last_value = value.detach().item()
        # guardar hc usado para esta decisión (detached para evitar crecer el grafo)
        self.last_hc = (self.hc[0].detach(), self.hc[1].detach())
        # actualizar LSTM y detach para el siguiente paso
        self.hc = (new_hc[0].detach(), new_hc[1].detach())

        return self.last_action

    # --------------------------------------------------------
    #                   RECIBIR ACCIÓN OPONENTE
    # --------------------------------------------------------
    def recibir_eleccion_del_oponente(self, eleccion: Elecciones):
        if self.last_action is None:
            self.historial.append(
                (Elecciones.COOPERAR, eleccion)
            )
            return

        # recompensa del dilema del prisionero
        if self.last_action == Elecciones.COOPERAR and eleccion == Elecciones.COOPERAR:
            r = 3.0
        elif self.last_action == Elecciones.COOPERAR and eleccion == Elecciones.TRAICIONAR:
            r = 0.0
        elif self.last_action == Elecciones.TRAICIONAR and eleccion == Elecciones.COOPERAR:
            r = 5.0
        else:
            r = 1.0

        self.historial.append((self.last_action, eleccion))

        # bootstrap del valor siguiente
        nuevo_estado_vec = encode_estado(self.historial, self.tamaño_estado)
        x2 = torch.tensor(nuevo_estado_vec, dtype=torch.float32, device=self.device)
        x2 = x2.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            _, next_value, _ = self.net(x2, self.hc)

        td_target = r + self.gamma * next_value.item()
        advantage = td_target - self.last_value

        # Si está congelado, no entrenar
        if self.frozen:
            self.last_action = None
            self.last_value = None
            self.last_state_vec = None
            self.last_hc = None
            return

        # entrenamiento
        s = torch.tensor(self.last_state_vec, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        logits, value_pred, _ = self.net(s, self.last_hc)

        log_probs = torch.log_softmax(logits, dim=-1)  # (1,2)
        probs = torch.softmax(logits, dim=-1)          # (1,2)
        action_idx = 0 if self.last_action == Elecciones.COOPERAR else 1

        # FIX: index with (batch, action) because logits are (1,2)
        action_logprob = log_probs[0, action_idx]

        # Entropía por acción, media en el batch
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        policy_loss = -action_logprob * advantage
        value_loss = (value_pred.squeeze() - td_target) ** 2

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.actual_loss = loss.item()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()

        self.last_action = None
        self.last_value = None
        self.last_state_vec = None
        self.last_hc = None

    # --------------------------------------------------------
    #              NUEVO OPONENTE (reset memoria LSTM)
    # --------------------------------------------------------
    def notificar_nuevo_oponente(self):
        # Controlar si se resetea historial y/o memoria LSTM entre oponentes
        if self.reset_history_on_new_opponent:
            self.historial = []
        self.last_action = None
        self.last_value = None
        self.last_state_vec = None
        self.last_hc = None

        if self.reset_memory_on_new_opponent:
            self.hc = self.net.init_hidden(1)
            self.hc = (self.hc[0].to(self.device), self.hc[1].to(self.device))


    def get_puntaje_acumulado(self) -> str:
        return "\033[95m" + f"{super().get_puntaje_acumulado()}" + "\033[0m"

    def get_puntaje_de_este_torneo(self) -> str:
        return "\033[95m" + f"{super().get_puntaje_de_este_torneo()}" + "\033[0m"

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
        self.net.eval()
        self.frozen = True
        for param in self.net.parameters():
            param.requires_grad = False

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