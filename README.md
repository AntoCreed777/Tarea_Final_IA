## 📋 Requisitos Previos

e:

- **Python 3.12+**  
- Tener instalado **pdm**, puedes consultar en https://pdm-project.org/en/latest/#__tabbed_1_2
- Puedes instalar los paquetes requeridos con `pdm install`, de otra forma puedes hacerlo manualmente según lo descrito en el archivo `pyproject.toml`

Para iniciar el torneo de todos contra todos:

Para elegir cual modelo elegir para competir basta con quitar el comentario respectivo. Ej: 

```bash
estrategias = [
        SiempreCoopera(),
        .
        .
        .
        Tullock(),
        # agente,
        A2C.load("QTables/A2C1.pt"),
        # A2C_LSTM.load("QTables/A2C_LSTM4.pt"),
        # DeepQNetwork.load("QTables/DeepQNetwork1.pt"),
        # DuelingDQN.load("QTables/DuelingDQN.pt")
    ]

```
Luego, ejecutar desde la raíz:

```bash
python3 -m src.main
```
Para iniciar el entrenamiento de los modelos ejecutar:

Es necesario cambiar manualmente el modelo a ejecutar en el archivo `src/arco_de_entrenamiento.py`. Bastaría con agregar el nombre de este a la lista de `protas` en la linea 157 junto a la métrica que se quiere guardar. Ej:


```python
  protas = [
    ["LSTM", Metrica.PERDIDA],
    ["A2C", Metrica.PERDIDA],
    ["QLearning", Metrica.EXPLORACION],
    ["SARSA", Metrica.EXPLORACION],
    ["DeepQNetwork", Metrica.PERDIDA],
    ["Dueling_dqn", Metrica.PERDIDA]
  ]

```
Se puede elegir entre los modelos: DeepQNetwork, A2C, LSTM, QLearning, SARSA, DeepQN y DuelingDQN.

Y luego correr desde la raíz:


```bash
python3 -m src.arco_de_entrenamiento

```

____ 

## Estrategias del Primer Torneo de Axelrod (1980)

Las estrategias implementadas, obtenidas del primer torneo de axelrod son las siguientes.

|# | Estrategia               | Autor(es)                                  | Identificador        |
|--|--------------------------|--------------------------------------------|----------------------|
|1 | Tit For Tat              | Anatol Rapoport                            | TitForTat            |
|2 | Tideman and Chieruzzi    | T. Nicolaus Tideman and Paula Chieruzzi    | TidemanAndChieruzzi  |
|3 | Nydegger                 | Rudy Nydegger                              | Nydegger             |
|4 | Grofman                  | Bernard Grofman                            | Grofman              |
|5 | Shubik                   | Martin Shubik                              | Shubik               |
|6 | Stein and Rapoport       | Stein and Anatol Rapoport                  | SteinAndRapoport     |
|7 | Grudger                  | James W. Friedman                          | Grudger              |
|8 | Davis                    | Morton Davis                               | Davis                |
|9 | Graaskamp                | Jim Graaskamp                              | Graaskamp            |
|10| FirstByDowning           | Leslie Downing                             | RevisedDowning       |
|11| Feld                     | Scott Feld                                 | Feld                 |
|12| Joss                     | Johann Joss                                | Joss                 |
|13| Tullock                  | Gordon Tullock                             | Tullock              |
|14| (Name withheld)          | Unknown                                    | UnnamedStrategy      |
|15| Random                   | Unknown                                    | Random               |

Visitar [la pagina web oficial](https://axelrod.readthedocs.io/en/fix-documentation/reference/overview_of_strategies.html) para más información acerca de su implementación.

