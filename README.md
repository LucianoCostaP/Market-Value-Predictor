#  MarketValue Predictor

Este proyecto estima el valor de mercado de jugadores de fútbol profesionales a partir de estadísticas históricas y transferencias reales. Usa datos de **FBref** y **Transfermarkt**, junto con un modelo automatizado entrenado con **PyCaret**.

##  Objetivos

- Scrapear estadísticas detalladas de jugadores desde FBref.
- Obtener transferencias históricas desde Transfermarkt.
- Construir un dataset realista de mercado de transferencias.
- Entrenar un modelo que prediga el valor de mercado de un jugador.
- Usar el modelo para estimar valores actuales para jugadores propios.

## Estructura del proyecto

project_root/
├── marketvalue/ # Módulo principal
│ ├── fetch.py # Obtención y parsing de datos de jugadores
│ ├── processing.py # Feature engineering
│ ├── model.py # Carga y predicción con el modelo entrenado
│ └── config.py # Constantes, rutas y configuración
│
├── scrapers/ # Scrapers y helpers auxiliares
│ ├── transfermarkt.py # Scraper de Transfermarkt
│ └── utils.py # Funciones auxiliares
│
├── scripts/ # Scripts ejecutables
│ ├── build_training_dataset.py # Construye el dataset de entrenamiento
│ ├── train_model.py # Entrena y guarda el modelo
│ └── predict_value.py # CLI para predicción
│
├── data/ # Datos crudos y procesados
│ ├── raw_stats/ # Stats originales por temporada
│ ├── transfers/ # Transferencias scrapeadas
│ ├── train_dataset.parquet # Dataset final para entrenamiento
│ └── produccion.csv # Datos actuales para predicción
│
├── models/ # Modelos entrenados (PyCaret)
│ └── transfer_model.pkl
│

