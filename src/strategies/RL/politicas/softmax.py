from src.elecciones import Elecciones
from src.strategies.RL.politicas.policy import Policy
import numpy as np

class Softmax(Policy):
    """
    Implementación de la politica SoftMax, la cual le da una probabilidad
    a cada acción según una temperatura, la temperatura aumenta tomar decisiones
    menos seguras.
    """

    def __init__(self, temperatura = 0.5):
        """
        Inicializa la temperatura de la politica.

        Args:
             temperatura: Temperatura que se usará en la formula de SoftMax
        """
        self.temperatura = temperatura

    def eleccion(self, q_table , estado ) -> Elecciones:
        """
        Decide la acción de forma aleatoria según las probabilidades
        que netrega la formula de softmax.
        """
        probs_dict = self._softmax(q_table[estado])

        acciones = list(probs_dict.keys())
        probs = list(probs_dict.values())

        # selección estocástica
        accion_elegida = np.random.choice(acciones, p=probs)

        return accion_elegida

    def _softmax(self, qvalues):
        """
        Funcion interna encargada de calcular la probablidad
        para cada decision.
        """
        acciones = list(qvalues.keys())
        valores = np.array(list(qvalues.values()), dtype=float)

            # estabilidad numérica
        valores_stable = valores - np.max(valores)
        exp_vals = np.exp(valores_stable / self.temperatura)
        probs = exp_vals / np.sum(exp_vals)

            # devolver diccionario acción → probabilidad
        return {accion: prob for accion, prob in zip(acciones, probs)}

