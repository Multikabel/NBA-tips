import pandas as pd
import time
import os
import joblib
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamelog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nba.com/',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}

# --- 1. INTELIGENTNÍ NAČÍTÁNÍ DAT ---
def stahni_data():
    # PRIORITA: Pokud existuje finální soubor nahraný ručně, použijeme ten
    if os.path.exists('nba_data_final.csv'):
        print("Nalezen nahraný soubor nba_data_final.csv. Používám jej pro výpočty...")
        df = pd.read_csv('nba_data_final.csv')
        
        # --- OPRAVA NÁZVŮ SLOUPCŮ (tady byla chyba) ---
        # Převedeme všechny názvy na velká písmena, aby to sedělo na TEAM_ID, PTS atd.
        df.columns = [c.upper() for c in df.columns]
        
        # Odstraníme staré vypočítané sloupce, aby se vytvořily nové
        cols_to_drop = ['ROLL_PTS_HOME', 'ROLL_PTS_AWAY', 'ELO_HOME', 'ELO_AWAY', 'HOME_WIN']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Pokud tam TEAM_ID stále není, zkusíme ho vytvořit z TEAM (častá varianta v exportu)
        if 'TEAM_ID' not in df.columns and 'TEAM' in df.columns:
            df = df.rename(columns={'TEAM': 'TEAM_ID'})
            
        return df

    # ZÁLOHA: Pokus o stažení
    print("Soubor nenalezen, zkouším stáhnout data z NBA API...")
    try:
        log = leaguegamelog.LeagueGameLog(season='2025-26', headers=HEADERS, timeout=60)
        return log.get_data_frames()[0]
    except Exception as e:
        print(f"Chyba při stahování: {e}")
        return pd.DataFrame()

# --- 2. TRANSFORMAČNÍ LOGIKA ---
def priprav_data(raw_df):
    if raw_df.empty:
        raise ValueError("Žádná data k dispozici!")
        
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
    
    # Vyhodíme řádky, kde ještě nemáme dost historie pro Rolling Stats
    return df.dropna(subset=['ROLL_PTS_HOME', 'ROLL_PTS_AWAY'])

# --- 3. ELO FUNKCE ---
def vypocitej_elo(df):
    # Inicializace Elo pro všechny týmy
    teams_list = pd.concat([df['TEAM_NAME_HOME'], df['TEAM_NAME_AWAY']]).unique()
    elo_dict = {team: 1500 for team in teams_list}
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
    print(f"Model i data úspěšně uloženy. Počet řádků: {len(df)}")

# --- SPOUŠTĚČ ---
if __name__ == "__main__":
    try:
        raw = stahni_data()
        processed = priprav_data(raw)
        final = vypocitej_elo(processed)
        trenuj_a_uloz(final)
        print("Vše proběhlo v pořádku!")
    except Exception as e:
        print(f"KRITICKÁ CHYBA: {e}")
