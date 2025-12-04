from src.elecciones import Elecciones
from src.strategies.base_class import base_strategies


class Shubik(base_strategies):
    """
    La estrategia Shubik coopera inicialmente y sigue cooperando hasta 
    que el oponente traiciona por primera vez. Cuando ocurre una 
    traición, responde con una secuencia de traiciones cuyo número 
    coincide con la cantidad total de veces que el oponente ha traicionado 
    hasta ese momento. Después de completar esa represalia, vuelve a 
    cooperar. Cada nueva traición del oponente incrementa en uno la 
    longitud de la próxima represalia.

    Las traiciones que ocurren durante el periodo de represalia no se cuentan.

    Atributos heredados:
        puntaje (int): Acumulador de recompensas definido en la clase base.

    Atributos:
        ultima_respuesta_oponente (Elecciones): Última acción observada del oponente.
        cantidad_de_traiciones (int): Total de traiciones del oponente registradas (fuera de represalia).
        contador_represalia (int): Número de rondas de traición pendientes como represalia.
    """
    
    def __init__(self):
        super().__init__()
        # Comienza cooperando.
        self.ultima_respuesta_oponente = Elecciones.COOPERAR
        self.cantidad_de_traiciones = 0
        self.contador_represalia = 0


    def realizar_eleccion(self) -> Elecciones:
        """
        Coopera salvo que esté ejecutando una represalia por una traición previa,
        en cuyo caso traiciona tantas veces como traiciones acumuladas haya.
        
        Returns:
            Elecciones: Cooperar, salvo durante la represalia.
        """
        
        if self.ultima_respuesta_oponente == Elecciones.TRAICIONAR and self.contador_represalia > 0: 
            self.contador_represalia -= 1
            return Elecciones.TRAICIONAR
        return Elecciones.COOPERAR

    def recibir_eleccion_del_oponente(self, eleccion: Elecciones) -> None:
        """
        Registra la elección del oponente. Si el oponente traiciona y no hay
        una represalia en curso, incrementa el contador de traiciones y ajusta
        el contador de represalias a ese total.
        
        Args:
            eleccion (Elecciones): Elección efectuada por el oponente.
        """
        
        self.ultima_respuesta_oponente = eleccion

        if eleccion == Elecciones.TRAICIONAR and self.contador_represalia == 0: 
            self.cantidad_de_traiciones += 1
            self.contador_represalia = self.cantidad_de_traiciones


    def notificar_nuevo_oponente(self) -> None:
        """
        Notifica el inicio de un enfrentamiento con un nuevo oponente.
        Resetea los contadores de traiciones y represalias y asume cooperación inicial.
        """
        
        self.contador_represalia = 0
        self.cantidad_de_traiciones = 0
        self.ultima_respuesta_oponente = Elecciones.COOPERAR

