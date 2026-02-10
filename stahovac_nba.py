import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamelog

def stahni_aktualni_data():
    # Dynamicky nastavíme sezónu podle aktuálního data (únor 2026 -> 2025-26)
    sezona_id = '2025-26' 
    
    print(f"--- Start stahování NBA dat pro sezónu {sezona_id} ---")

    try:
        # LeagueGameLog stáhne VŠECHNY zápasy celé ligy jedním požadavkem
        # Je to mnohem bezpečnější pro GitHub Actions (nehrozí Rate Limit)
        log = leaguegamelog.LeagueGameLog(
            season=sezona_id,
            season_type_all_star='Regular Season'
        )
        df = log.get_data_frames()[0]
        
        if not df.empty:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # Uložíme jako nba_data_raw.csv (aby to sedělo na tvůj YAML)
            file_name = 'nba_data_raw.csv'
            df.to_csv(file_name, index=False)
            
            print(f"Úspěch! Staženo {len(df)} řádků.")
            print(f"Soubor uložen v: {os.path.abspath(file_name)}")
        else:
            print("API vrátilo prázdnou tabulku. Zkontroluj ID sezóny.")

    except Exception as e:
        print(f"Kritická chyba při stahování: {e}")

if __name__ == "__main__":
    import os
    stahni_aktualni_data()
