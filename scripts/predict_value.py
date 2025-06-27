import sys
import os

ROOT_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PROJECT_PATH)

import pandas as pd
from pathlib import Path

from pycaret.regression import load_model, predict_model

from marketvalue.config import (
    PRODUCTION_DATA_PATH,
    MODEL_PATH,
)
from marketvalue.feature_engineering import build_dataset


def main(player_id: int):
    # Cargar datos de producción
    df = pd.read_csv(PRODUCTION_DATA_PATH)

    # Filtrar por player_id
    fila = df[df["player_id"] == player_id]
    if fila.empty:
        print(f"No se encontró el jugador con ID {player_id}")
        return

    fila = fila.iloc[[0]]  # Solo la primera fila si hay duplicados

    # Procesamiento de features (sin reentrenar PCA/encoders)
    X, Y = build_dataset(
        fila,
        train=False,
    )

    # Cargar modelo ya entrenado
    model = load_model(MODEL_PATH)

    # Realizar predicción
    resultado = predict_model(model, data=X)

    # Mostrar fee estimado

    fee_estimado = resultado.prediction_label.iloc[0]
    print(f"Valor de mercado estimado: €{fee_estimado:,.0f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python scripts/predict.py <player_id>")
        sys.exit(1)

    player_id = int(sys.argv[1])
    main(player_id)