import sys
import os

ROOT_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PROJECT_PATH)

from marketvalue.config import ROOT_DIR, RAW_STATS_DIR, TRANSFERS_DIR
# Añadir la raíz del proyecto al PYTHONPATH
sys.path.append(str(ROOT_DIR))

import uuid
import unidecode
import pandas as pd
import LanusStats as ls
from scrape.fbref import get_stats_league, concat_stats
from scrape.id_transfermarkt import agregar_player_ids
from scrape.transfers_transfermarkt import get_all_transfers

def valid_ids(stats, stats_path):
    tm = ls.Transfermarkt()
    fake_names_id = []
    repeated = stats.groupby("player_id")["Player"].nunique()
    repeated = repeated[repeated > 1]
    stats_repeated = stats[stats["player_id"].isin(repeated.index)]
    ids_repeated = stats_repeated.player_id.unique()

    for id in ids_repeated:
        real_name = tm.get_player_market_value(player_id=int(id))["player"].iloc[0]
        names_repeated_id = stats.loc[stats.player_id == id, "Player"].unique()
        for name in names_repeated_id:
            if unidecode.unidecode(name) != unidecode.unidecode(real_name):
                fake_names_id.append((name, id))

    # Construir un DataFrame con los pares a eliminar para usar merge con indicador
    if fake_names_id:
        df_fake = pd.DataFrame(fake_names_id, columns=["Player", "player_id"])
        # Filtrar filas que NO estén en df_fake
        before_count = len(stats)
        type_stats = stats.player_id.dtype
        df_fake.player_id = df_fake.player_id.astype(type_stats)
        stats = stats.merge(df_fake.assign(to_drop=True), on=["Player", "player_id"], how="left")
        stats = stats[stats["to_drop"].isna()].drop(columns=["to_drop"])
        after_count = len(stats)
        print(f"Eliminadas {before_count - after_count} filas con player_id y Player no coincidentes.")
    else:
        print("No se encontraron filas a eliminar.")

    stats.to_csv(stats_path, index=False)
    return stats


def merge_incremental_stats(new_stats, stats_path):
    if stats_path.exists():
        df_existente = pd.read_csv(stats_path)
        temp_new_stats_path = stats_path.parent / f"temp_new_stats_{uuid.uuid4().hex}.csv"
        new_stats.to_csv(temp_new_stats_path, index=False)
        new_stats = pd.read_csv(temp_new_stats_path)
        temp_new_stats_path.unlink()
    else:
        df_existente = pd.DataFrame()

    df_existente = df_existente.reset_index(drop=True)
    new_stats = new_stats.reset_index(drop=True)

    df_actualizado = pd.concat([df_existente, new_stats], ignore_index=True)

    columnas_clave = ["Player", "Season", "stats_Squad"]
    df_actualizado = df_actualizado.drop_duplicates(subset=columnas_clave, keep="last")

    df_actualizado.to_csv(stats_path, index=False)
    print(f"Guardado dataset actualizado con {len(df_actualizado)} filas en {stats_path}")

    return df_actualizado


def agregar_player_id(stats_updated, stats_path):
    stats_with_ids = agregar_player_ids(stats_updated)
    stats_with_ids.to_csv(stats_path, index=False)
    print(f"Stats con player_id guardados en {stats_path}")
    return stats_with_ids

def obtener_transferencias(stats_with_ids, trmkt, transfers_dir, tipo="field_players"):
    transfers_path = transfers_dir / f"all_transfers_{tipo}.csv"

    # Cargar transferencias existentes si hay
    if transfers_path.exists():
        existing_transfers = pd.read_csv(transfers_path)
        existing_transfers["player_id"] = existing_transfers["player_id"].astype("Int64")
        existing_ids = set(existing_transfers["player_id"].dropna().unique())
        print(f"Transferencias existentes: {len(existing_transfers)} jugadores")
    else:
        existing_transfers = pd.DataFrame()
        existing_ids = set()
        print("No hay transferencias previas, creando nuevo archivo")

    # Filtrar stats con player_id válido y tipo seguro
    stats_with_ids["player_id"] = pd.to_numeric(stats_with_ids["player_id"], errors="coerce")
    stats_validos = stats_with_ids[stats_with_ids["player_id"].notna()]
    stats_validos["player_id"] = stats_validos["player_id"].astype("Int64")


    # Detectar nuevos jugadores
    nuevos = stats_validos[~stats_validos["player_id"].isin(existing_ids)]
    print(f"Nuevos jugadores con transferencias por buscar: {len(nuevos)}")

    if nuevos.empty:
        print("No hay jugadores nuevos. Nada que hacer.")
        return

    # Obtener transferencias nuevas
    nuevas_transferencias = get_all_transfers(nuevos, trmkt, max_workers=16)

    if nuevas_transferencias is not None and not nuevas_transferencias.empty:
        nuevas_transferencias["player_id"] = nuevas_transferencias["player_id"].astype("Int64")
        df_actualizado = pd.concat([existing_transfers, nuevas_transferencias], ignore_index=True)
        df_actualizado.to_csv(transfers_path, index=False)
        print(f"Guardadas {len(nuevas_transferencias)} nuevas transferencias en {transfers_path}")
    else:
        print("No se obtuvieron nuevas transferencias.")


def main():
    leagues = ["Saudi League"]
    seasons = ['2021-2022', '2023-2024', '2022-2023', '2024-2025']

    fbref = ls.Fbref()
    trmkt = ls.Transfermarkt()

    RAW_STATS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)

    all_new_stats_field_players = []
    all_new_stats_goalkeepers = []
    for league in leagues:
        for season in seasons:
            print(f"Scrapeando stats de {league} temporada {season}...")
            stats_field_players, stats_goalkeepers = get_stats_league(fbref, league, season)
            all_new_stats_field_players.append(stats_field_players)
            all_new_stats_goalkeepers.append(stats_goalkeepers)

    stats_concat_field_players = concat_stats(all_new_stats_field_players, goalkeepers=False)
    stats_concat_goalkeepers = concat_stats(all_new_stats_goalkeepers, goalkeepers=True)

    stats_path_field_players = RAW_STATS_DIR / "all_stats_field_players.csv"
    stats_path_goalkeepers = RAW_STATS_DIR / "all_stats_goalkeepers.csv"

    # Merge incremental: agrego nuevas stats a las que ya estaban
    stats_updated_field_players = merge_incremental_stats(stats_concat_field_players, stats_path_field_players)
    stats_updated_goalkeepers = merge_incremental_stats(stats_concat_goalkeepers, stats_path_goalkeepers)

    # Agregar player_id y sobrescribir el mismo archivo
    stats_with_ids_field_players =  agregar_player_id(stats_updated_field_players, stats_path_field_players)
    stats_with_ids_goalkeepers = agregar_player_id(stats_updated_goalkeepers, stats_path_goalkeepers)

    stats_with_ids_field_players = valid_ids(stats_with_ids_field_players, stats_path_field_players)
    stats_with_ids_goalkeepers = valid_ids(stats_with_ids_goalkeepers, stats_path_goalkeepers)

    # Obtener transferencias
    obtener_transferencias(stats_with_ids_field_players, trmkt, TRANSFERS_DIR)
    obtener_transferencias(stats_with_ids_goalkeepers, trmkt, TRANSFERS_DIR, tipo = "goalkeepers")

if __name__ == "__main__":
    main()
