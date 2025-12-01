Para correr el proyecto, ejecutar desde la raíz:

```bash
python3 -m src.main
```

Se recomienda usar `black` e `isort` para mantener la claridad del código. Para instalarlos, usar:

```bash
pipx install black isort
```

Luego, para formatear el código:

```bash
black .
isort .
```
