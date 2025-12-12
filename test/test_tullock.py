import pytest
from src.strategies.originales.tullock import Tullock
from src.elecciones import Elecciones

def test_initial_cooperation_first_11_rounds():
    t = Tullock()
    # For the first 11 rounds, realizar_eleccion must return COOPERAR.
    for _ in range(11):
        assert t.realizar_eleccion() == Elecciones.COOPERAR
        # simulate end of round: opponent defects (advances ronda and fills history)
        t.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)


def test_defect_after_11_with_zero_probability(monkeypatch):
    t = Tullock()
    # Simulate 11 rounds where opponent always defects -> last 10 are all TRAICIONAR
    for _ in range(11):
        t.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)

    # prob_cooperar should be 0.0, so any random should cause TRAICIONAR
    monkeypatch.setattr("src.strategies.originales.tullock.random.random", lambda: 0.0)
    assert t.realizar_eleccion() == Elecciones.TRAICIONAR

    monkeypatch.setattr("src.strategies.originales.tullock.random.random", lambda: 0.999)
    assert t.realizar_eleccion() == Elecciones.TRAICIONAR


def test_probabilistic_cooperation_when_random_below_prob(monkeypatch):
    t = Tullock()
    # Build 11 opponent choices such that the last 10 contain 8 cooperations and 2 defections
    # Example sequence (11 items): [TRAICIONAR, COOP x8, TRAICIONAR, TRAICIONAR]
    seq = [
        Elecciones.TRAICIONAR,
        Elecciones.COOPERAR, Elecciones.COOPERAR, Elecciones.COOPERAR,
        Elecciones.COOPERAR, Elecciones.COOPERAR, Elecciones.COOPERAR,
        Elecciones.COOPERAR, Elecciones.COOPERAR,
        Elecciones.TRAICIONAR,
        Elecciones.TRAICIONAR,
    ]
    for e in seq:
        t.recibir_eleccion_del_oponente(e)

    # Now last 10 have 8 cooperations -> prop_cooperar = 0.8 -> prob_cooperar = 0.7
    # If random() < 0.7, should COOPERAR
    monkeypatch.setattr("src.strategies.originales.tullock.random.random", lambda: 0.6)
    assert t.realizar_eleccion() == Elecciones.COOPERAR

    # If random() == 0.7 (not less than), should TRAICIONAR
    monkeypatch.setattr("src.strategies.originales.tullock.random.random", lambda: 0.7000000000001)
    assert t.realizar_eleccion() == Elecciones.TRAICIONAR


def test_random_equal_to_prob_results_in_defection(monkeypatch):
    t = Tullock()
    # Create last 10 with 8 cooperations again -> prob_cooperar = 0.7
    seq = [Elecciones.TRAICIONAR] + [Elecciones.COOPERAR] * 8 + [Elecciones.TRAICIONAR] + [Elecciones.TRAICIONAR]
    # Only need 11 receives; ensure we call exactly 11 times
    for e in seq[:11]:
        t.recibir_eleccion_del_oponente(e)

    monkeypatch.setattr("src.strategies.originales.tullock.random.random", lambda: 0.7000001)
    assert t.realizar_eleccion() == Elecciones.TRAICIONAR


def test_notificar_nuevo_oponente_resets_state():
    t = Tullock()
    # simulate some rounds
    for _ in range(5):
        t.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    assert t.ronda == 5
    assert len(t._historial_oponente) > 0

    # reset for new opponent
    t.notificar_nuevo_oponente()
    assert t.ronda == 0
    assert len(t._historial_oponente) == 0
    # after reset, should immediately cooperate
    assert t.realizar_eleccion() == Elecciones.COOPERAR