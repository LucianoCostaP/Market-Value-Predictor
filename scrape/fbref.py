import pandas as pd

def get_stats_league(fbref, league, season):
    field_players, goalkeepers = fbref.get_all_player_season_stats(league, season, save_csv=False)
    field_players["Season"] = season
    goalkeepers["Season"] = season
    return field_players, goalkeepers

def concat_stats(stats, goalkeepers = False):
    stats_concat = pd.concat(stats, ignore_index = True)
    if goalkeepers:
        stats_concat.rename(columns={'keepers_Born': 'stats_Born', 'keepers_Squad': 'stats_Squad'}, inplace=True)
    stats_concat.stats_Born = stats_concat.stats_Born.astype(int)
    return stats_concat