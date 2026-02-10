import pandas as pd
import time
import os
import joblib
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. STAHOVÁNÍ DAT ---
from nba_api.stats.endpoints import leaguegamelog
import time

# Definujeme hlavičky, aby nás API nevykoplo
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive',
}

def stahni_data(sezony=['2024-25', '2025-26']):
    seznam_zapasu = []
    print(f"--- Start stahování s Headers ---")

    for sezona_id in sezony:
        retries = 5 # Zvýšíme počet pokusů
        while retries > 0:
            try:
                print(f"Stahuji {sezona_id}...", end=" ", flush=True)
                
                # PŘIDÁVÁME HEADERS A TIMEOUT
                log = leaguegamelog.LeagueGameLog(
                    season=sezona_id,
                    season_type_all_star='Regular Season',
                    headers=HEADERS,
                    timeout=60  # Prodloužíme čekání na 60 sekund
                )
                
                df_sezona = log.get_data_frames()[0]
                if not df_sezona.empty:
                    seznam_zapasu.append(df_sezona)
                    print(f"OK ({len(df_sezona)} řádků)")
                    time.sleep(5) # Delší pauza mezi sezónami, abychom je nenaštvali
                    break
            except Exception as e:
                retries -= 1
                print(f"Chyba: {e}. Zkouším znovu za 10s...")
                time.sleep(10)
    
    # ... zbytek funkce (concat atd.)

    full_df = pd.concat(seznam_zapasu).drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])
    return full_df

# --- 2. TRANSFORMAČNÍ LOGIKA ---
def priprav_data(raw_df):
    raw_df['GAME_DATE'] = pd.to_datetime(raw_df['GAME_DATE'])
    raw_df = raw_df.sort_values('GAME_DATE')

    # Rolling stats na úrovni týmu (forma posledních 10 zápasů)
    raw_df['ROLL_PTS'] = raw_df.groupby('TEAM_ID')['PTS'].transform(lambda x: x.rolling(10, closed='left').mean())
    
    # Home/Away Merge
    home = raw_df[raw_df['MATCHUP'].str.contains('vs.')].copy()
    away = raw_df[raw_df['MATCHUP'].str.contains('@')].copy()
    
    home = home.rename(columns=lambda x: x + '_HOME' if x not in ['GAME_ID', 'GAME_DATE'] else x)
    away = away.rename(columns=lambda x: x + '_AWAY' if x not in ['GAME_ID', 'GAME_DATE'] else x)
    
    df = pd.merge(home, away, on=['GAME_ID', 'GAME_DATE'])
    df['HOME_WIN'] = (df['WL_HOME'] == 'W').astype(int)
    return df.dropna(subset=['ROLL_PTS_HOME', 'ROLL_PTS_AWAY'])

# --- 3. TVOJE ELO FUNKCE ---
def vypocitej_elo(df):
    elo_dict = {team: 1500 for team in pd.concat([df['TEAM_NAME_HOME'], df['TEAM_NAME_AWAY']]).unique()}
    K = 20
    elo_home_list, elo_away_list = [], []

    for _, row in df.iterrows():
        h_team, a_team = row['TEAM_NAME_HOME'], row['TEAM_NAME_AWAY']
        current_h_elo, current_a_elo = elo_dict[h_team], elo_dict[a_team]
        
        elo_home_list.append(current_h_elo)
        elo_away_list.append(current_a_elo)
        
        exp_h = 1 / (1 + 10 ** ((current_a_elo - current_h_elo) / 400))
        actual_h = 1 if row['HOME_WIN'] == 1 else 0
        
        elo_dict[h_team] = current_h_elo + K * (actual_h - exp_h)
        elo_dict[a_team] = current_a_elo - K * (actual_h - exp_h)

    df['ELO_HOME'], df['ELO_AWAY'] = elo_home_list, elo_away_list
    return df

# --- 4. TRÉNINK MODELU ---
def trenuj_a_uloz(df):
    features = ['ROLL_PTS_HOME', 'ROLL_PTS_AWAY', 'ELO_HOME', 'ELO_AWAY']
    X = df[features]
    y = df['HOME_WIN']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, 'nba_model.pkl')
    df.to_csv('nba_data_final.csv', index=False)
    print("Model i data úspěšně uloženy.")

# --- SPOUŠTĚČ ---
if __name__ == "__main__":
    raw = stahni_data()
    processed = priprav_data(raw)
    final = vypocitej_elo(processed)
    trenuj_a_uloz(final)
