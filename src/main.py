from src.controlador_duelos import ControladorDuelos
from src.strategies import (Davis, Downing, Feld, Grofman, Random,
                            SiempreCoopera, SiempreTraiciona, TitForTat)

if __name__ == "__main__":

    estrategias = [
        SiempreCoopera(),
        SiempreTraiciona(),
        TitForTat(),
        Random(),
        Davis(),
        Downing(),
        Feld(),
        Grofman(),
    ]

    torneo = ControladorDuelos(
        estrategias,
        cantidad_de_torneos=10,
        jugadas_base_duelo=500,
        limite_de_variacion_de_jugadas=50,
    )

    torneo.iniciar_duelos()

    print("\nRESULTADOS")
    for e in estrategias:
        print(type(e).__name__, "→", "{:,}".format(e.puntaje))  # Con separador de miles
