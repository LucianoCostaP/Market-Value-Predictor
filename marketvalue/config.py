from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Carpetas de datos
RAW_STATS_DIR = ROOT_DIR / "data" / "raw_stats"
TRANSFERS_DIR = ROOT_DIR / "data" / "transfers"
PLAYERS_IDS_PATH = ROOT_DIR / "data" / "players" / "players_ids.csv"

STATS_FIELD_PLAYERS_PATH = RAW_STATS_DIR / "all_stats_field_players.csv"
STATS_GOALKEEPERS_PATH = RAW_STATS_DIR / "all_stats_goalkeepers.csv"
STATS_NEWS_PATH = RAW_STATS_DIR / "player_news_stats.csv"

TRANSFERS_FIELD_PLAYERS_PATH = TRANSFERS_DIR / "all_transfers_field_players.csv"
TRANSFERS_GOALKEEPERS_PATH = TRANSFERS_DIR / "all_transfers_goalkeepers.csv"

PRODUCTION_DATA_PATH = ROOT_DIR / "data" / "production_data.csv"
TRAINING_DATA_PATH = ROOT_DIR / "data" / "train_dataset.csv"

MODEL_PATH = ROOT_DIR / "models" / "transfer_model"  # sin extensión, PyCaret maneja .pkl

# Carpeta para guardar encoders y transformadores
ENCODERS_DIR = ROOT_DIR / "encoders"
ENCODERS_DIR.mkdir(parents=True, exist_ok=True)  # asegurar que exista

MLB_DIR = ENCODERS_DIR / "mlb"       # Carpeta para guardar MultiLabelBinarizer por columna
MLB_DIR.mkdir(parents=True, exist_ok=True)

# Rutas específicas para guardar objetos
MLB_DICT_PATH = ENCODERS_DIR / "mlb_dict.joblib"
CT_ENCODER_PATH = ENCODERS_DIR / "ct_encoder.joblib"
TARGET_ENCODER_PATH = ENCODERS_DIR / "target_encoder.joblib"

# Carpeta para guardar PCA por grupo
PCA_ENCODER_DIR = ENCODERS_DIR / "pca_groups"
PCA_ENCODER_DIR.mkdir(parents=True, exist_ok=True)  # asegurar que exista
PCA_GROUPS_PATH = PCA_ENCODER_DIR / "pca_groups.joblib"

SEASON = "2024-2025"

