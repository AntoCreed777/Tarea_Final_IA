import types
import pytest
from src.strategies.originales.stein_rapoport import SteinRapoport
from src.elecciones import Elecciones

import src.strategies.originales.stein_rapoport as stein_mod


def advance_rounds(agent: SteinRapoport, n: int, reward: int = 0):
    for _ in range(n):
        agent.recibir_recompensa(reward)


def test_initial_state():
    s = SteinRapoport()
    assert s.ultima_respuesta_oponente == Elecciones.COOPERAR
    assert s.ronda == 0
    # puntaje should start at 0 (inherited from base class)
    assert getattr(s, "puntaje", 0) == 0
    assert s.cooperaciones_oponente == 0
    assert s.traiciones_oponente == 0
    assert s.oponente_es_random is False


def test_first_five_rounds_cooperate():
    s = SteinRapoport()
    # According to the implementation, rounds 0..4 (ronda <= 4) return COOPERAR
    for _ in range(5):
        assert s.realizar_eleccion() == Elecciones.COOPERAR
        s.recibir_recompensa(0)


def test_tft_behavior_after_initial_rounds():
    s = SteinRapoport()
    # advance to ronda 5
    advance_rounds(s, 5)
    # set opponent last move to TRAICIONAR
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    # for rounds 5..12 (ronda < 13) the strategy returns ultima_respuesta_oponente
    for _ in range(8):
        assert s.realizar_eleccion() == Elecciones.TRAICIONAR
        s.recibir_recompensa(0)


def test_chi_square_sets_random_and_forces_defection(monkeypatch):
    s = SteinRapoport()
    # prepare some observed counts
    s.cooperaciones_oponente = 10
    s.traiciones_oponente = 5
    # ensure alpha attribute exists (code uses self.alpha which is a bug; set it for test)
    s.alpha = 0.05

    # patch the chisquare name in the module to return an object with .pvalue
    class FakeResult:
        def __init__(self, p):
            self.pvalue = p

    def fake_chisq(counts):
        # return pvalue greater than alpha to mark opponent as random
        return FakeResult(0.1)

    monkeypatch.setattr(stein_mod, "chisquare", fake_chisq)

    # advance to ronda 15 so that self.ronda % 15 == 0 branch is taken
    advance_rounds(s, 15)
    # ensure ultima respuesta is cooperate so we can see the forced defection
    s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)

    choice = s.realizar_eleccion()
    # since fake chisq yields pvalue >= alpha, oponente_es_random should be True and strategy defects
    assert s.oponente_es_random is True
    assert choice == Elecciones.TRAICIONAR


def test_chi_square_non_random_keeps_tft(monkeypatch):
    s = SteinRapoport()
    s.cooperaciones_oponente = 8
    s.traiciones_oponente = 7
    s.alpha = 0.05

    class FakeResult:
        def __init__(self, p):
            self.pvalue = p

    def fake_chisq_low(counts):
        # return pvalue lower than alpha to mark opponent as non-random
        return FakeResult(0.01)

    monkeypatch.setattr(stein_mod, "chisquare", fake_chisq_low)

    advance_rounds(s, 15)
    # set opponent last move to COOPERAR, expect strategy to mirror when not random
    s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    choice = s.realizar_eleccion()
    assert s.oponente_es_random is False
    assert choice == Elecciones.COOPERAR


def test_notificar_nuevo_oponente_and_recibir_recompensa_effects():
    s = SteinRapoport()
    # change some internals
    s.ultima_respuesta_oponente = Elecciones.TRAICIONAR
    s.cooperaciones_oponente = 3
    s.traiciones_oponente = 2
    s.oponente_es_random = True
    s.puntaje = 10
    s.ronda = 7

    s.notificar_nuevo_oponente()
    assert s.ultima_respuesta_oponente == Elecciones.COOPERAR
    assert s.ronda == 0
    assert s.oponente_es_random is False
    assert s.cooperaciones_oponente == 0
    assert s.traiciones_oponente == 0

    # recibir_recompensa should add to puntaje and increment ronda
    prev_puntaje = s.puntaje
    prev_ronda = s.ronda
    s.recibir_recompensa(5)
    assert s.puntaje == prev_puntaje + 5
    assert s.ronda == prev_ronda + 1