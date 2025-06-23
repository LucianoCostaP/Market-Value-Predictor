
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_STATS_DIR = ROOT_DIR / "data" / "raw_stats"
TRANSFERS_DIR = ROOT_DIR / "data" / "transfers"
PLAYERS_IDS_PATH = ROOT_DIR / "data" / "players" / "players_ids.csv"

STATS_FIELD_PLAYERS_PATH = RAW_STATS_DIR / "all_stats_field_players.csv"
STATS_GOALKEEPERS_PATH = RAW_STATS_DIR / "all_stats_goalkeepers.csv"

TRANSFERS_FIELD_PLAYERS_PATH = TRANSFERS_DIR / "all_transfers_field_players.csv"
TRANSFERS_GOALKEEPERS_PATH = TRANSFERS_DIR / "all_transfers_goalkeepers.csv"

PRODUCTION_DATA_PATH = ROOT_DIR / "data" / "production_data.csv"
TRAINING_DATA_PATH = ROOT_DIR / "data" / "train_dataset.csv"

MODEL_PATH = ROOT_DIR / "models" / "transfer_model.pkl"

SEASON = "2024-2025"