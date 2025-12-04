import pytest
from src.strategies.originales.shubik import Shubik
from src.elecciones import Elecciones

def test_initial_cooperates():
    s = Shubik()
    assert s.realizar_eleccion() == Elecciones.COOPERAR


def test_cooperates_when_opponent_cooperates_repeatedly():
    s = Shubik()
    for _ in range(5):
        s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
        assert s.realizar_eleccion() == Elecciones.COOPERAR


def test_single_betrayal_triggers_one_retaliation_then_cooperate():
    s = Shubik()
    # Round 1: opponent betrays -> Shubik should retaliate once
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    assert s.realizar_eleccion() == Elecciones.TRAICIONAR
    
    # Round 2: after retaliation, no outstanding represalia -> cooperate
    # simulate opponent cooperating now
    s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    assert s.realizar_eleccion() == Elecciones.COOPERAR


def test_betrayals_during_represalia_are_not_counted():
    s = Shubik()
    # Opponent betrays -> sets cantidad_de_traiciones to 1 and contador_represalia to 1
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    assert s.cantidad_de_traiciones == 1
    assert s.contador_represalia == 1

    # Before Shubik responds, opponent betrays again (this should NOT be counted because we're in represalia)
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    # counts should be unchanged
    assert s.cantidad_de_traiciones == 1
    assert s.contador_represalia == 1

    # Now Shubik responds: performs the one retaliation
    assert s.realizar_eleccion() == Elecciones.TRAICIONAR
    assert s.contador_represalia == 0


def test_multiple_betrayals_increase_future_represalias():
    s = Shubik()
    sequence = []
    # Round 1: opponent betrays -> new count = 1, Shubik retaliates (1)
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    sequence.append(s.realizar_eleccion())

    # Round 2: opponent betrays again (since contador_represalia == 0 this becomes the 2nd counted betrayal)
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    sequence.append(s.realizar_eleccion())  # retaliation 1 of 2

    # Round 3: opponent betrays again (during the 2-reprisal period this should not increase count)
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    sequence.append(s.realizar_eleccion())  # retaliation 2 of 2

    # After finishing the two reprisal moves, next should be cooperate
    s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    sequence.append(s.realizar_eleccion())

    assert sequence == [
        Elecciones.TRAICIONAR,  # after first betrayal
        Elecciones.TRAICIONAR,  # first of two retaliations (after second counted betrayal)
        Elecciones.TRAICIONAR,  # second of two retaliations
        Elecciones.COOPERAR,    # back to cooperation
    ]


def test_notificar_nuevo_oponente_resets_state():
    s = Shubik()
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    assert s.cantidad_de_traiciones >= 1
    s.notificar_nuevo_oponente()
    assert s.cantidad_de_traiciones == 0
    assert s.contador_represalia == 0
    assert s.ultima_respuesta_oponente == Elecciones.COOPERAR
    assert s.realizar_eleccion() == Elecciones.COOPERAR