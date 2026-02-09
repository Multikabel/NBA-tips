import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder

def stahni_sezonu_bezpecne(sezona_id='2024-25'):
    vsechny_tymy = teams.get_teams()
    seznam_zapasu = []

    print(f"--- Start stahování sezóny {sezona_id} ---")

    for i, tym in enumerate(vsechny_tymy):
        try:
            print(f"[{i+1}/30] Stahuji: {tym['full_name']}...", end=" ", flush=True)
            
            finder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=tym['id'],
                season_nullable=sezona_id,
                league_id_nullable='00'
            )
            
            tym_zapas = finder.get_data_frames()[0]
            seznam_zapasu.append(tym_zapas)
            print("OK")
            
            # Trochu delší pauza pro jistotu
            time.sleep(1.5) 
            
        except Exception as e:
            print(f"CHYBA u {tym['full_name']}: {e}")
            time.sleep(5) # Při chybě počkáme déle
            continue

    if seznam_zapasu:
        df = pd.concat(seznam_zapasu).reset_index(drop=True)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        # Uložíme hned, jakmile máme alespoň něco
        df.to_csv('nba_data_raw.csv', index=False)
        print(f"\nHotovo! Uloženo {len(df)} řádků do 'nba_data_raw.csv'.")
    else:
        print("\nNepodařilo se stáhnout žádná data.")

stahni_sezonu_bezpecne()