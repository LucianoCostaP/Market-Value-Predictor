from marketvalue.config import (
    STATS_FIELD_PLAYERS_PATH,
    STATS_GOALKEEPERS_PATH,
    TRANSFERS_FIELD_PLAYERS_PATH,
    TRANSFERS_GOALKEEPERS_PATH,
    SEASON,
    PRODUCTION_DATA_PATH,
    TRAINING_DATA_PATH,
)

import pandas as pd
from marketvalue.processing import prepare_data

def main():
    stats_field_players = pd.read_csv(STATS_FIELD_PLAYERS_PATH)
    transfers_field_players = pd.read_csv(TRANSFERS_FIELD_PLAYERS_PATH)

    data_training, data_production = prepare_data(SEASON, stats_field_players, transfers_field_players)
    

    #Ahora transformar data_training y data_production usando 
    #Ingeniería de características que se van a utilizar el modelo
    #Revisar que hacer con los jugadores que fueron traspasados a mitad de temporada
    #La columna fee de stats_field_players es el target


    data_production.to_csv(PRODUCTION_DATA_PATH, index=False)
    data_training.to_csv(TRAINING_DATA_PATH, index = False)

if __name__ == "__main__":
    main()
