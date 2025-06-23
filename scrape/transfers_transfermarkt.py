from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

def get_all_transfers(player_stats_df, trmkt, max_workers=16):
    players_stats = player_stats_df[player_stats_df['player_id'].notnull()]
    players_stats.player_id = players_stats.player_id.astype(int)
    players_stats.player_id = players_stats.player_id.astype("object")
    player_ids = players_stats["player_id"].dropna().unique()
    transfer_histories = []

    def fetch_and_tag(pid):
        df = trmkt.get_player_transfer_history(player_id=pid)
        if df is not None and not df.empty:
            df["player_id"] = pid
            return df
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_tag, pid): pid for pid in player_ids}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching transfers"):
            try:
                result = future.result()
                if result is not None:
                    transfer_histories.append(result)
            except Exception as e:
                pid = futures[future]
                print(f"Error fetching player_id {pid}: {e}")

    if transfer_histories:
        return pd.concat(transfer_histories, ignore_index=True)
    else:
        return pd.DataFrame()