import pytest
from src.strategies.nydegger import Nydegger
from src.elecciones import Elecciones

def test_tft_first_moves_basic():
    g = Nydegger()
    # Ronda 1: inicio coopera, oponente traiciona -> CT
    assert g.realizar_eleccion() == Elecciones.COOPERAR
    
    g.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)

    # Ronda 2: Tit for Tat -> debe traicionar (replicar la última del oponente)
    assert g.realizar_eleccion() == Elecciones.TRAICIONAR
    # Completar segunda ronda: oponente traiciona -> TT
    g.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)

    # Ronda 3: según TFT debe traicionar
    assert g.realizar_eleccion() == Elecciones.TRAICIONAR


def test_third_move_exception_per_spec():
    """
    Según la especificación en el docstring: si las dos primeras rondas fueron CT (yo cooperé, oponente traicionó)
    y luego TC (yo traicioné, oponente cooperó), entonces en la tercera ronda debe traicionar.
    Este test verifica esa regla (puede fallar si la implementación tiene la excepción mal verificada).
    """
    g = Nydegger()
    # Ronda 1 -> CT
    g.realizar_eleccion()  # cooperó inicialmente
    g.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    # Ronda 2 -> TFT => traiciona, luego oponente coopera => TC
    assert g.realizar_eleccion() == Elecciones.TRAICIONAR
    g.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    # Ronda 3 -> según especificación, debe traicionar
    assert g.realizar_eleccion() == Elecciones.TRAICIONAR


def test_historial_encoding_and_A_logic():
    g = Nydegger()
    # Preparar un historial explícito de tres resultados (orden cronológico: antiguo -> reciente)
    # Queremos A = 1 (que está en el conjunto de defectos).
    # A = r3*16 + r2*4 + r1*1  donde r1 es el más antiguo.
    # Para A == 1 -> r3=0, r2=0, r1=1  => historial = [1, 0, 0]
    g.historial_respuestas.clear()
    g.historial_respuestas.append(1)
    g.historial_respuestas.append(0)
    g.historial_respuestas.append(0)
    assert len(g.historial_respuestas) == 3
    assert g.realizar_eleccion() == Elecciones.TRAICIONAR

    # Caso contrario: todos cooperaron -> A = 0 -> debe cooperar
    g.historial_respuestas.extend([0, 0, 0])
    assert g.realizar_eleccion() == Elecciones.COOPERAR


def test_recibir_eleccion_encodings():
    g = Nydegger()
    g.historial_respuestas.clear()

    # CC -> 0
    g.ultima_respuesta_propia = Elecciones.COOPERAR
    g.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    assert list(g.historial_respuestas)[-1] == 0

    # TC -> 1 (yo traicioné, oponente cooperó)
    g.ultima_respuesta_propia = Elecciones.TRAICIONAR
    g.recibir_eleccion_del_oponente(Elecciones.COOPERAR)
    assert list(g.historial_respuestas)[-1] == 1

    # CT -> 2 (yo cooperé, oponente traicionó)
    g.ultima_respuesta_propia = Elecciones.COOPERAR
    g.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    assert list(g.historial_respuestas)[-1] == 2

    # TT -> 3 (ambos traicionaron)
    g.ultima_respuesta_propia = Elecciones.TRAICIONAR
    g.recibir_eleccion_del_oponente(Elecciones.TRAICIONAR)
    assert list(g.historial_respuestas)[-1] == 3