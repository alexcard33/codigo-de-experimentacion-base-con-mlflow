# TC5061 · Experimentación con MLflow — Costos de seguro médico

**Autor:** Manuel Alejandro Cardona Esquivel | A01795954

Script de entrenamiento parametrizado e instrumentado con **MLflow Tracking** para predecir el costo anual del seguro
médico (`charges`, USD) del dataset *Medical Cost Personal* (1,338 registros). Parte del modelo base (regresión lineal)
de la actividad anterior y lo compara contra variantes de Random Forest.

## Estructura

```
├── train.py              # carga → limpieza → división → entrenamiento → evaluación → registro en MLflow
├── run_experiments.sh    # las 9 corridas comparadas en el reporte
├── requirements.txt      # dependencias con versiones fijas
├── data/insurance.csv    # dataset
└── notebooks/
    └── 01_eda.ipynb      # solo exploración y decisiones de limpieza (sin entrenamiento)
```

## Instalación

Requiere Python 3.11 o superior.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Con los valores por defecto (Random Forest, 100 árboles, profundidad 4, semilla 42):

```bash
python train.py
```

Cambiando hiperparámetros sin editar el código:

```bash
python train.py --n-estimators 300 --max-depth 5 --min-samples-leaf 5 --seed 42
```

Baseline lineal de la actividad anterior:

```bash
python train.py --model linear
```

Todas las corridas del reporte:

```bash
bash run_experiments.sh
```

### Argumentos

| Argumento | Defecto | Descripción |
|---|---|---|
| `--model` | `random_forest` | `random_forest` o `linear` |
| `--n-estimators` | `100` | Número de árboles |
| `--max-depth` | `4` | Profundidad máxima de cada árbol (`0` = sin límite) |
| `--min-samples-leaf` | `1` | Mínimo de registros por hoja |
| `--max-features` | `1.0` | Fracción de variables evaluadas en cada división |
| `--test-size` | `0.2` | Proporción del conjunto de prueba |
| `--cv-folds` | `5` | Particiones de validación cruzada sobre entrenamiento (`0` la desactiva) |
| `--seed` | `42` | Semilla aleatoria |
| `--data-path` | `data/insurance.csv` | Ruta del dataset |
| `--experiment-name` | `TC5061_Costos_Seguro_Medico` | Experimento de MLflow |
| `--tracking-uri` | `sqlite:///mlflow.db` | Almacén de MLflow o URL de un servidor |
| `--run-name` | automático | Nombre de la corrida |

`python train.py --help` muestra la misma ayuda.

## Qué registra cada corrida en MLflow

- **Parámetros:** modelo, hiperparámetros, `test_size`, `cv_folds`, `seed`, tamaño de train/test y `data_md5` (huella del CSV).
- **Métricas:** `test_rmse`, `test_mae`, `test_r2`, las mismas en train, y `cv_rmse_mean` / `cv_rmse_std`.
- **Artefactos** (`evaluacion/`): `real_vs_predicho.png`, `importancia_variables.png` y `metrics.json`.
- **Modelo:** el `Pipeline` completo (preprocesamiento + estimador) con firma de entrada/salida, en `model/`.

## Ver los resultados

En otra terminal, desde la carpeta del repositorio:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir http://127.0.0.1:5000 → experimento **TC5061_Costos_Seguro_Medico** → seleccionar las corridas → **Compare**.

Si ya tienes un servidor de MLflow levantado, apunta el script a él:
`python train.py --tracking-uri http://127.0.0.1:5000`.

## Reproducibilidad

- La semilla (`--seed`) controla la división train/test, los pliegues de validación cruzada y el Random Forest;
  además se fijan `random`, `numpy` y `PYTHONHASHSEED`.
- Las versiones de todas las dependencias están fijadas en `requirements.txt`.
- La huella `data_md5` registrada permite verificar que se usó el mismo archivo de datos.
- Comprobación: `run_experiments.sh` repite la configuración por defecto (`repro_check_defaults`) y obtiene las
  mismas métricas que la corrida original hasta el undécimo decimal (`test_rmse = 4261.00137231859…`). La diferencia
  en la última cifra (~1e-12) viene del orden de la suma en paralelo de los árboles (`n_jobs=-1`), no de la semilla.

## Dataset

*Medical Cost Personal Dataset* (Kaggle), originalmente publicado con el libro *Machine Learning with R* (B. Lantz).
Copia usada: <https://github.com/stedy/Machine-Learning-with-R-datasets>.
