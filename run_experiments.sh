#!/usr/bin/env bash
# Corre todas las configuraciones comparadas en el reporte, en el mismo experimento de MLflow.
# Uso: bash run_experiments.sh
set -euo pipefail

python train.py --model linear                                                    # 1. baseline de la actividad anterior
python train.py                                                                   # 2. Random Forest con valores por defecto
python train.py --max-depth 2                                                     # 3. árboles muy poco profundos
python train.py --n-estimators 200 --max-depth 0                                  # 4. árboles sin límite de profundidad
python train.py --n-estimators 200 --max-depth 0 --min-samples-leaf 10            # 5. sin límite pero con hojas grandes
python train.py --n-estimators 500                                                # 6. más árboles, misma profundidad
python train.py --max-features 0.5                                                # 7. menos variables por división
python train.py --n-estimators 300 --max-depth 5 --min-samples-leaf 5             # 8. configuración afinada
python train.py --run-name repro_check_defaults                                   # 9. repite la 2 para probar reproducibilidad
