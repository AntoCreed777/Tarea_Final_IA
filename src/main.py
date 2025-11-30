from src.controlador_duelos import ControladorDuelos
from src.strategies import (Davis, Random, SiempreCoopera, SiempreTraiciona,
                            TitForTat)

if __name__ == "__main__":

    estrategias = [
        SiempreCoopera(),
        SiempreTraiciona(),
        TitForTat(),
        Random(),
        Davis(),
    ]

    torneo = ControladorDuelos(
        estrategias,
        cantidad_de_torneos=100,
        jugadas_base_duelo=5000,
        limite_de_variacion_de_jugadas=500,
    )

    torneo.iniciar_duelos()

    print("\nRESULTADOS")
    for e in estrategias:
        print(type(e).__name__, "→", "{:,}".format(e.puntaje))  # Con separador de miles
