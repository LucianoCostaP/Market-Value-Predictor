import LanusStats as ls
import re
import pandas as pd
from collections import defaultdict

def parse_transfer_type_and_fee(value):
    transfer_type = "Desconocida"
    fee_value = 0.0

    if isinstance(value, (int, float)):
        return "Transferencia", float(value)

    if not isinstance(value, str):
        return "Desconocida", value

    if "Fin de cesión" in value:
        return "Fin de cesión", 0.0

    if "Libre" in value:
        return "Libre", 0.0

    if "Coste de cesión" in value:
        transfer_type = "Cesión"
        # Buscar dentro del HTML el número
        match = re.search(r'Coste de cesión:.*?([\d,.]+)\s*(mil|mill)\. €', value)
        if match:
            num_str, unit = match.groups()
            try:
                num = float(num_str.replace(",", "."))
                multiplier = 1_000_000 if unit == "mill" else 1_000
                fee_value = num * multiplier
            except:
                fee_value = 0.0
        return transfer_type, fee_value

    if "Cesión" in value:
        return "Cesión", 0.0

    # Valor en millones o miles
    match = re.search(r'([\d,.]+)\s*(mil|mill)\. €', value)
    if match:
        num_str, unit = match.groups()
        try:
            num = float(num_str.replace(",", "."))
            multiplier = 1_000_000 if unit == "mill" else 1_000
            return "Transferencia", num * multiplier
        except:
            return "Transferencia", value

    return transfer_type, value

def expand_season(season_str):
    try:
        start, end = season_str.split("/")
        start_full = "20" + start.zfill(2)
        end_full = "20" + end.zfill(2)
        return f"{start_full}-{end_full}"
    except:
        return season_str

def parse_fee(value):
    if isinstance(value, str):
        value = value.strip()
        if "mill. €" in value:
            num = value.replace("mill. €", "").replace(".", "").replace(",", ".").strip()
            try:
                return float(num) * 1_000_000
            except ValueError:
                return value
        elif "mil €" in value:
            num = value.replace("mil €", "").replace(".", "").replace(",", ".").strip()
            try:
                return float(num) * 1_000
            except ValueError:
                return value
    return value

def previous_season(season_str):
    try:
        start, end = season_str.split("-")
        prev_start = int(start) - 1
        prev_end = int(end) - 1
        return f"{prev_start}-{prev_end}"
    except:
        return None


def calcular_dias_en_club(transfers: pd.DataFrame) -> pd.DataFrame:
    # Asegurar formato de fecha
    transfers["transfer_date"] = pd.to_datetime(transfers["dateUnformatted"], errors="coerce")

    # Ordenar por jugador y fecha de transferencia
    transfers = transfers.sort_values(["player_id", "transfer_date"]).reset_index(drop=True)

    # Inicializar columna
    transfers["days_in_club"] = pd.NA

    # Recorrer por jugador
    for pid, group in transfers.groupby("player_id"):
        prev_transfer = None
        prev_club_to = None

        for idx, row in group.iterrows():
            current_date = row["transfer_date"]
            current_club_from = row["club_from"]

            if (
                prev_transfer is not None
                and prev_club_to == current_club_from
                and pd.notna(current_date)
            ):
                days = (current_date - prev_transfer["transfer_date"]).days
                transfers.at[idx, "days_in_club"] = days

            # Actualizar referencia para la próxima iteración
            prev_transfer = row
            prev_club_to = row["club_to"]

    return transfers


def transform_age(age):
    age = str(age)
    if '-' in age:
        age = int(age.split("-")[0])
    else:
        age = int(age)
    return age

def remove_duplicate_suffix_columns(df):

    suffix_groups = defaultdict(list)

    for col in df.columns:
        if "_" in col:
            suffix = col.split("_", 1)[1]
            suffix_groups[suffix].append(col)

    # Eliminar todas menos una por grupo
    cols_to_drop = []
    for group in suffix_groups.values():
        if len(group) > 1:
            # Mantener la última columna por defecto
            cols_to_drop.extend(group[:-1])

    return df.drop(columns=cols_to_drop)

def get_all_transfers(transfers):
    transfers["fee"] = transfers["fee"].map(parse_fee)
    transfers = transfers.dropna(subset=["dateUnformatted"])
    transfers[["transfer_type", "fee"]] = transfers["fee"].apply(
        lambda x: pd.Series(parse_transfer_type_and_fee(x))
        )
    transfers["season"] = transfers["season"].apply(expand_season)
    transfers["season_start"] = transfers["season"].str[:4].astype(int)
    transfers = transfers.loc[(transfers["season_start"] >= 2021) & 
        (transfers.transfer_type == "Transferencia")]
    transfers = transfers[["player_id", "season", "fee"]]
    transfers["prev_season"] = transfers["season"].apply(previous_season)
    transfers["prev_season_start"] = transfers["prev_season"].str[:4]
    transfers = transfers[["player_id", "prev_season_start", "fee"]]
    return transfers

def remove_null_ids(df, column_id):
    df = df[df[column_id].notna()]
    return df

def join_transfers_stats(transfers, stats):
    transfers_stats = transfers.merge(stats, 
        left_on=["player_id", "prev_season_start"],
        right_on=["player_id", "season_start"],
        how="inner",
        suffixes=("", "_prev_stats"))
    return transfers_stats

def prepare_stats(stats):
    stats = remove_duplicate_suffix_columns(stats)
    stats.misc_Age = stats.misc_Age.transform(transform_age)
    stats.Season = stats.Season.astype(str)
    stats = stats.drop_duplicates(keep='first')
    return stats


def prepare_data(season, stats, transfers, news):
    years = season.split("-")
    second_year = years[1]

    stats = remove_null_ids(stats, "player_id")
    stats = prepare_stats(stats)
    
    stats_produccion = stats.loc[(stats.Season == season) | (stats.Season == second_year)]
    stats_produccion.player_id = stats_produccion.player_id.astype(int)

    stats = stats.loc[~((stats.Season == season) | (stats.Season == second_year))]
    stats["season_start"] = stats["Season"].str[:4]
    
    transfers = calcular_dias_en_club(transfers)
    transfers = get_all_transfers(transfers)

    transfers_stats = join_transfers_stats(transfers, stats)
    transfers_stats.drop(["prev_season_start", "season_start"], inplace = True, axis = 1)
    news_cols = [col for col in news.columns if col != 'player_id']

    transfers_stats = pd.merge(transfers_stats, news, on='player_id', how='left')
    transfers_stats[news_cols] = transfers_stats[news_cols].fillna(0)

    stats_produccion = pd.merge(stats_produccion, news, on='player_id', how='left')
    stats_produccion[news_cols] = stats_produccion[news_cols].fillna(0)
    return transfers_stats, stats_produccion
