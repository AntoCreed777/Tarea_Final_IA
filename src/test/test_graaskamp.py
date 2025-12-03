import pytest
from src.strategies.graaskamp import Graaskamp
from src.elecciones import Elecciones

def test_initial_cooperate_and_51_betray_then_tft_window():
    g = Graaskamp()

    # Initial decision (ronda == 0) should cooperate
    assert g.realizar_eleccion() == Elecciones.COOPERAR

    # Simulate 51 opponent cooperations to advance rounds to 51
    for _ in range(51):
        g.recibir_eleccion_del_oponente(Elecciones.COOPERAR)

    # At ronda == 51 the strategy must betray once
    assert g.realizar_eleccion() == Elecciones.TRAICIONAR

    # Next 5 rounds (52..56) should play TFT (i.e., cooperate given the opponent cooperated)
    for _ in range(52, 57):
        g.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
        assert g.realizar_eleccion() == Elecciones.COOPERAR


def test_detects_random_opponent_and_always_betrays_after_detection():
    g = Graaskamp()
    g.notificar_nuevo_oponente()

    # Force the internal state to a late round and balanced counts so chisquare yields high p-value
    g.ronda = 60
    g.cooperaciones_oponente = 30
    g.traiciones_oponente = 30

    # Should detect as random and betray
    assert g._chequear_tipo_oponente()
    assert g.realizar_eleccion() == Elecciones.TRAICIONAR

    # Remain betraying when considered random
    g.cooperaciones_oponente = 15
    g.traiciones_oponente = 15
    assert g.realizar_eleccion() == Elecciones.TRAICIONAR


def test_opponent_clone_detection_preserved_when_histories_match():
    g = Graaskamp()
    # Start with opponent cooperating
    g.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    first_response = g.realizar_eleccion()
    assert first_response == Elecciones.COOPERAR

    # Simulate an opponent that mirrors our previous move for several rounds
    for _ in range(5):
        # opponent plays exactly what we last played (clone behavior)
        g.recibir_eleccion_del_oponente(g.ultimo_movimiento_propio)
        resp = g.realizar_eleccion()
        # Our response should match last opponent move (TFT behavior) and histories remain aligned
        assert resp == g.ultimo_movimiento_oponente

    # The implementation should still consider the opponent a clone given identical recent histories
    assert g.oponente_es_clon is True