import pytest
from src.strategies.originales.tideman_chieruzzi import TidemanChieruzzi
from src.elecciones import Elecciones

def test_initial_choice_is_cooperate():
    s = TidemanChieruzzi()
    assert s.realizar_eleccion() == Elecciones.COOPERAR
    # initial internal flags
    assert s.ultima_respuesta_propia == Elecciones.COOPERAR
    assert s.ultima_respuesta_oponente == Elecciones.COOPERAR


def test_single_defection_triggers_one_retaliation():
    s = TidemanChieruzzi()
    # first round: strategy would cooperate
    assert s.realizar_eleccion() == Elecciones.COOPERAR
    # opponent defects
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    assert s.cantidad_de_traiciones == 1
    assert s.cantidad_de_traiciones_recordadas == 1
    assert s.contador_represalias == 1
    # next action should be a single retaliation
    assert s.realizar_eleccion() == Elecciones.TRAICIONAR
    # counter decremented after using reprisal
    assert s.contador_represalias == 0


def test_sequential_rounds_accumulate_traiciones_recordadas_and_set_reprisals():
    s = TidemanChieruzzi()
    # round 1
    assert s.realizar_eleccion() == Elecciones.COOPERAR
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)  # recordadas -> 1, reprisal set to 1
    assert s.cantidad_de_traiciones_recordadas == 1
    assert s.contador_represalias == 1
    # strategy retaliates this round
    assert s.realizar_eleccion() == Elecciones.TRAICIONAR
    assert s.contador_represalias == 0
    # opponent defects again in a later round -> recordadas increments and reprisal becomes 2
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    assert s.cantidad_de_traiciones_recordadas == 2
    assert s.contador_represalias == 2
    # two retaliations expected now
    assert s.realizar_eleccion() == Elecciones.TRAICIONAR
    assert s.realizar_eleccion() == Elecciones.TRAICIONAR
    assert s.contador_represalias == 0


def test_puntaje_oponente_updates_for_all_outcomes():
    s = TidemanChieruzzi()
    # own last action cooperate, opponent cooperate -> C_C[1] == 3
    s.ultima_respuesta_propia = Elecciones.COOPERAR
    s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    assert s.puntaje_oponente == 3

    # own cooperate, opponent defect -> C_T[1] == 5 (3 + 5 = 8)
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    assert s.puntaje_oponente == 8

    # set own last action to defect, opponent cooperate -> T_C[1] == 0 (no change)
    s.ultima_respuesta_propia = Elecciones.TRAICIONAR
    s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    assert s.puntaje_oponente == 8  # +0
    
    # defect-defect -> T_T[1] == 1
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    assert s.puntaje_oponente == 9


def test_validar_nueva_oportunidad_all_conditions_lead_to_cooperate_and_reset():
    s = TidemanChieruzzi()
    # simulate many rounds passed since last opportunity
    s.ronda = 21
    s.ultima_nueva_oportunidad = 0

    # set large lead in our favor
    s.puntaje = 100
    s.puntaje_oponente = 0

    # opponent currently cooperating
    s.ultima_respuesta_oponente = Elecciones.COOPERAR

    # make N large and set cantidad_de_traiciones_recordadas far from 50-50 range
    s.cantidad_de_cooperaciones = 1
    s.cantidad_de_traiciones = 20
    s.cantidad_de_traiciones_recordadas = 20  

    # calling realizar_eleccion should detect valid "nueva oportunidad" and return COOPERAR
    choice = s.realizar_eleccion()
    assert choice == Elecciones.COOPERAR
    # and state must be reset for a new opportunity
    assert s.cantidad_de_traiciones_recordadas == 0
    assert s.contador_represalias == 0
    assert s.ultima_nueva_oportunidad == s.ronda
    assert s.ultima_respuesta_propia == Elecciones.COOPERAR


def test_notificar_nuevo_oponente_resets_state():
    s = TidemanChieruzzi()
    # mutate internal state
    s.cantidad_de_traiciones = 5
    s.cantidad_de_cooperaciones = 7
    s.contador_represalias = 3
    s.puntaje = 42
    s.puntaje_oponente = 10
    s.ronda = 100
    s.ultima_respuesta_propia = Elecciones.TRAICIONAR
    s.ultima_respuesta_oponente = Elecciones.TRAICIONAR

    # reset
    s.notificar_nuevo_oponente()
    assert s.cantidad_de_traiciones == 0
    assert s.cantidad_de_cooperaciones == 0
    assert s.contador_represalias == 0
    assert s.puntaje == 42  # puntaje is not reset by notificar_nuevo_oponente in implementation
    assert s.puntaje_oponente == 0
    assert s.ronda == 0
    assert s.ultima_respuesta_propia == Elecciones.COOPERAR
    assert s.ultima_respuesta_oponente == Elecciones.COOPERAR


def test_duel_length_increments_each_round():
    s = TidemanChieruzzi()
    # start at round 0
    assert s.ronda == 0
    # round 1: our move + opponent response
    assert s.realizar_eleccion() in (Elecciones.COOPERAR, Elecciones.TRAICIONAR)
    s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    s.recibir_recompensa(3)
    assert s.ronda == 1
    # round 2
    assert s.realizar_eleccion() in (Elecciones.COOPERAR, Elecciones.TRAICIONAR)
    s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    s.recibir_recompensa(1)
    assert s.ronda == 2
    # round 3
    assert s.realizar_eleccion() in (Elecciones.COOPERAR, Elecciones.TRAICIONAR)
    s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    s.recibir_recompensa(1)
    assert s.ronda == 3


def test_largo_duelo_updates_as_average_of_previous_duels():
    s = TidemanChieruzzi()

    # Duel 1: 3 rounds
    for _ in range(3):
        assert s.realizar_eleccion() in (Elecciones.COOPERAR, Elecciones.TRAICIONAR)
        s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
        s.recibir_recompensa(3)
    assert s.ronda == 3
    s.notificar_nuevo_oponente()
    assert s.largo_duelo == 3  # average of [3]

    # Duel 2: 5 rounds
    for _ in range(5):
        assert s.realizar_eleccion() in (Elecciones.COOPERAR, Elecciones.TRAICIONAR)
        s.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
        s.recibir_recompensa(1)
    assert s.ronda == 5
    s.notificar_nuevo_oponente()
    assert s.largo_duelo == (3 + 5) / 2  # average of [3, 5]

    # Duel 3: 2 rounds
    for _ in range(2):
        assert s.realizar_eleccion() in (Elecciones.COOPERAR, Elecciones.TRAICIONAR)
        s.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
        s.recibir_recompensa(3)
    assert s.ronda == 2
    s.notificar_nuevo_oponente()
    assert s.largo_duelo == (3 + 5 + 2) / 3  # average of [3, 5, 2]

# Alias to satisfy runner nodeid with the misspelling
def test_single_defection_triggers_one_retalation():
    test_single_defection_triggers_one_retaliation()

