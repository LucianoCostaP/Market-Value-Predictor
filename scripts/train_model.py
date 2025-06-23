from marketvalue.config import TRAINING_DATA_PATH, MODEL_PATH
import pandas as pd

def main():
    df = pd.read_csv(TRAINING_DATA_PATH)

    #Dividir la data en train, valid y test y usar Pycaret para evaluar el modelo
    #Guardar el modelo 
    #También hacer un notebook con el análisis del modelo

    #save_model(best_model, str(MODEL_PATH).replace(".pkl", ""))
