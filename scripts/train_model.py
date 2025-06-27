import sys
import os

ROOT_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PROJECT_PATH)

from marketvalue.config import (
    STATS_FIELD_PLAYERS_PATH,
    TRANSFERS_FIELD_PLAYERS_PATH,
    SEASON,
    PRODUCTION_DATA_PATH,
    TRAINING_DATA_PATH,
    MODEL_PATH,
)

import pandas as pd
from marketvalue.processing import prepare_data
from marketvalue.feature_engineering import build_dataset
from marketvalue.feature_engineering import train_pycaret_model

def main():
    stats_field_players = pd.read_csv(STATS_FIELD_PLAYERS_PATH)
    transfers_field_players = pd.read_csv(TRANSFERS_FIELD_PLAYERS_PATH)

    data_training, data_production = prepare_data(SEASON, stats_field_players, transfers_field_players)
    
    data_training, target = build_dataset(data_training)
    data_training["fee"] = target
    
    train_pycaret_model(data_training, "fee", MODEL_PATH)

    data_production.to_csv(PRODUCTION_DATA_PATH, index=False)
    data_training.to_csv(TRAINING_DATA_PATH, index = False)

if __name__ == "__main__":
    main()
