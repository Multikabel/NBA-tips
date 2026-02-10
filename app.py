import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # <--- TADY JE TA OPRAVA

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="NBA AI Predictor 2026", layout="wide", page_icon="🏀")
# ... zbytek kódu

# --- NAČTENÍ DAT A MODELU ---
@st.cache_data
def load_data():
    df = pd.read_csv('nba_data_final.csv')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

def get_latest_stats(df, team_name):
    """
    Najde poslední známé Elo a Rolling Stats pro tým bez ohledu na to, 
    jestli hrál naposledy doma nebo venku.
    """
    # Vyfiltrujeme všechny zápasy týmu (doma i venku)
    team_matches = df[(df['TEAM_NAME_HOME'] == team_name) | (df['TEAM_NAME_AWAY'] == team_name)]
    last_match = team_matches.sort_values('GAME_DATE').iloc[-1]
    
    if last_match['TEAM_NAME_HOME'] == team_name:
        return {
            'ELO': last_match['ELO_HOME'],
            'ROLLING_PTS': last_match['ROLLING_PTS_HOME']
        }
    else:
        return {
            'ELO': last_match['ELO_AWAY'],
            'ROLLING_PTS': last_match['ROLLING_PTS_AWAY']
        }

# --- HLAVNÍ LOGIKA ---
try:
    df = load_data()
    
    # Místo trénování v aplikaci doporučuji načíst ten .pkl z GitHubu
    # Je to rychlejší a aplikace se nebude sekat
    model_win = joblib.load('nba_model.pkl') 
    # (Pokud ho nemáš, necháme tvůj trénovací kód níže)

    # --- TVŮJ TRÉNOVACÍ BLOK (ponechán, pokud nepoužiješ .pkl) ---
    # @st.cache_resource
    # def train_all_models(data):
    #    features = ['ROLL_PTS_HOME', 'ROLL_PTS_AWAY', 'ELO_HOME', 'ELO_AWAY']
    #    X = data[features]
    #    model_win = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, data['HOME_WIN'])
    #    model_pts_h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, data['PTS_HOME'])
    #    model_pts_a = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, data['PTS_AWAY'])
    #    return model_win, model_pts_h, model_pts_a, features

    # model_win, model_pts_h, model_pts_a, features = train_all_models(df)

    # --- SIDEBAR VÝBĚR ---
    teams_list = sorted(df['TEAM_NAME_HOME'].unique())
    home_team = st.sidebar.selectbox("🏠 Domácí tým", teams_list)
    away_team = st.sidebar.selectbox("🚀 Hostující tým", teams_list, index=1)

    if home_team != away_team:
        # ZÍSKÁNÍ SKUTEČNĚ AKTUÁLNÍCH DAT
        stats_h = get_latest_stats(df, home_team)
        stats_a = get_latest_stats(df, away_team)

        input_df = pd.DataFrame([[
            stats_h['ROLLING_PTS'], 
            stats_a['ROLLING_PTS'], 
            stats_h['ELO'], 
            stats_a['ELO']
        ]], columns=features)

        # VÝPOČTY (tvoje logika)
        prob_home = model_win.predict_proba(input_df)[0][1]
        pred_h_pts = model_pts_h.predict(input_df)[0]
        pred_a_pts = model_pts_a.predict(input_df)[0]

        # --- UI DISPLAY (viz tvůj kód) ---
        # ... (zde pokračuje tvoje hezké UI se sloupci a progress bary)
        st.success(f"Analýza pro zápas {home_team} vs {away_team} připravena.")

except Exception as e:
    st.error(f"Chyba: {e}")
