import pytest
import pandas as pd
import enum
from unittest.mock import Mock, patch, MagicMock

from src.arco_de_entrenamiento import Metrica, torneos_de_entrenamiento

class TestTorneosDeEntrenamiento:
    
    def test_torneos_retorna_dataframe(self):
        """Verifica que la función retorna un DataFrame"""
        estrategia_mock = Mock()
        estrategia_mock.__class__.__name__ = "TestEstrategia"
        estrategia_mock.puntaje_torneo_actual = 100
        estrategia_mock.obtener_perdida_promedio.return_value = 0.5
        
        resultado = torneos_de_entrenamiento(
            estrategia_mock, 
            [Metrica.PERDIDA],
            episodios_por_torneo=10,
            cantidad_de_torneos=20,
            guardado_cada_n_torneos=10
        )
        
        assert isinstance(resultado, pd.DataFrame)
    
    def test_torneos_guarda_metricas_perdida(self):
        """Verifica que se guarden las métricas de pérdida correctamente"""
        estrategia_mock = Mock()
        estrategia_mock.__class__.__name__ = "TestEstrategia"
        estrategia_mock.puntaje_torneo_actual = 100
        estrategia_mock.obtener_perdida_promedio.return_value = 0.3
        
        resultado = torneos_de_entrenamiento(
            estrategia_mock,
            [Metrica.PERDIDA],
            cantidad_de_torneos=10,
            guardado_cada_n_torneos=5
        )
        
        assert "perdida" in resultado.columns
        assert resultado["perdida"].iloc[0] == 0.3
    
    def test_torneos_guarda_metricas_exploracion(self):
        """Verifica que se guarden las métricas de exploración correctamente"""
        estrategia_mock = Mock()
        estrategia_mock.__class__.__name__ = "TestEstrategia"
        estrategia_mock.puntaje_torneo_actual = 100
        estrategia_mock.obtener_porcentaje_exploracion.return_value = 0.45
        
        resultado = torneos_de_entrenamiento(
            estrategia_mock,
            [Metrica.EXPLORACION],
            cantidad_de_torneos=10,
            guardado_cada_n_torneos=5
        )
        
        assert "exploracion" in resultado.columns
    
    def test_torneos_puntaje_acumulado(self):
        """Verifica que el puntaje acumulado se incremente correctamente"""
        estrategia_mock = Mock()
        estrategia_mock.__class__.__name__ = "TestEstrategia"
        estrategia_mock.puntaje_torneo_actual = 50
        estrategia_mock.obtener_perdida_promedio.return_value = 0.1
        
        resultado = torneos_de_entrenamiento(
            estrategia_mock,
            [Metrica.PERDIDA],
            cantidad_de_torneos=20,
            guardado_cada_n_torneos=10
        )
        
        assert resultado["puntaje_acumulado"].iloc[-1] == 100
    
    def test_torneos_elimina_estrategia_duplicada(self):
        """Verifica que se elimine la estrategia del listado de oponentes si ya existe"""
        estrategia_mock = Mock()
        estrategia_mock.__class__.__name__ = "TitForTat"
        estrategia_mock.puntaje_torneo_actual = 75
        
        with patch('ControladorDuelos') as mock_controlador:
            torneos_de_entrenamiento(
                estrategia_mock,
                [],
                cantidad_de_torneos=1,
                guardado_cada_n_torneos=1
            )
            
            # Verificar que TitForTat fue removido de la lista
            assert mock_controlador.call_count >= 1