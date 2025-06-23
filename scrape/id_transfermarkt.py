import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import concurrent.futures
import unidecode
import os
import pandas as pd
import re
import functools
from marketvalue.config import PLAYERS_IDS_PATH

def name_normalize(name):
    name = str(name).lower()
    name = unidecode.unidecode(name)
    name = name.replace("-", " ")
    name = ''.join(c for c in name if c.isalnum() or c.isspace())
    name = ' '.join(name.split())
    return name

def search_transfermarkt_player_id(player_name, birth_year=None):
    player_name = name_normalize(player_name)

    base_url = "https://www.transfermarkt.com"
    search_url = f"{base_url}/schnellsuche/ergebnis/schnellsuche?query={player_name.replace(' ', '+')}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.transfermarkt.com/",
    }

    response = requests.get(search_url, headers=headers)
    time.sleep(3)

    if response.status_code != 200:
        print(f"Error {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    rows = soup.select("table.items > tbody > tr")

    current_year = time.localtime().tm_year

    for row in rows:
        name_link = row.select_one("td.hauptlink a")
        age_td = row.select("td.zentriert")[2] if len(row.select("td.zentriert")) > 2 else None

        if not name_link or not name_link.get("href") or not age_td:
            continue

        href = name_link["href"]
        age_text = age_td.get_text(strip=True)

        try:
            age = int(age_text)
            candidate_birth_year = current_year - age
        except:
            continue

        if birth_year is not None and abs(candidate_birth_year - birth_year) > 1:
            continue  # fuera del rango aceptable

        match = re.search(r"/spieler/(\d+)", href)
        if match:
            player_id = match.group(1)
            name = name_link.get("title") or name_link.text.strip()
            return player_id

    return None

def try_search_variants(player_name, birth_year):
    player_id = search_transfermarkt_player_id(player_name, birth_year)
    if player_id:
        return player_id
    apellido = player_name.split()[-1]
    if apellido != player_name:
        player_id = search_transfermarkt_player_id(apellido, birth_year)
        if player_id:
            return player_id
    nombre = player_name.split()[0]
    if nombre != player_name:
        player_id = search_transfermarkt_player_id(nombre, birth_year)
        if player_id:
            return player_id
    return None

def buscar_id_para_par(player_name, birth_year, repeated = False):
    if repeated:
        player_id = try_search_variants(player_name, birth_year)
        return player_name, birth_year, player_id
    player_id = search_transfermarkt_player_id(player_name, birth_year)
    return player_name, birth_year, player_id

def agregar_player_ids(stats, max_workers=5, save_every=50, repeated = False):
    # Definir rutas usando pathlib
    PLAYERS_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_csv = PLAYERS_IDS_PATH
    
    # Cargar resultados existentes si hay
    if os.path.exists(output_csv):
        df_existente = pd.read_csv(output_csv)
        df_existente = df_existente.drop_duplicates(subset=["Player", "stats_Born"], keep="last")
        mask_valid_id = df_existente['player_id'].notnull() & (df_existente['player_id'] != '')
        df_filtrado = df_existente.loc[mask_valid_id]
        ya_busque = set(zip(df_filtrado["Player"], df_filtrado["stats_Born"]))
    else:
        df_existente = pd.DataFrame(columns=["Player", "stats_Born", "player_id"])
        ya_busque = set()

    # Extraer pares únicos y filtrar los que ya fueron buscados
    pares_unicos = stats[['Player', 'stats_Born']].drop_duplicates()
    pares_faltantes = [tuple(x) for x in pares_unicos.to_numpy() if tuple(x) not in ya_busque]

    print(f"Total jugadores únicos: {len(pares_unicos)}")
    print(f"Faltan buscar: {len(pares_faltantes)}")

    acumulador = []
    procesados = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        buscar_id = functools.partial(buscar_id_para_par, repeated=repeated)
        futures = {
            executor.submit(buscar_id, player, birth): (player, birth)
            for player, birth in pares_faltantes
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Buscando player_id"):
            player_name, birth_year = futures[future]
            try:
                _, _, player_id = future.result()
            except Exception as e:
                print(f"Error con {player_name} ({birth_year}): {e}")
                player_id = None

            acumulador.append((player_name, birth_year, player_id))
            procesados += 1

            if procesados % save_every == 0:
                df_temp = pd.DataFrame(acumulador, columns=["Player", "stats_Born", "player_id"])
                df_existente = pd.concat([df_existente, df_temp])
                df_existente = df_existente.drop_duplicates(subset=["Player", "stats_Born"], keep="last")
                df_existente.to_csv(output_csv, index=False)
                acumulador = []

    # Guardar lo que quedó sin guardar
    if acumulador:
        df_temp = pd.DataFrame(acumulador, columns=["Player", "stats_Born", "player_id"])
        df_existente = pd.concat([df_existente, df_temp])
        df_existente = df_existente.drop_duplicates(subset=["Player", "stats_Born"], keep="last")
        df_existente.to_csv(output_csv, index=False)

    # Unir todos los resultados con el dataset original
    if "player_id" in stats.columns:
        stats = stats.drop(columns=["player_id"])
    stats = stats.merge(df_existente, on=["Player", "stats_Born"], how="left")

    return stats