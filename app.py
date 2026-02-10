import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # <--- TADY JE TA OPRAVA

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="NBA AI Predictor 2026", layout="wide", page_icon="🏀")
# ... zbytek kódu

# --- NAČTENÍ DAT A MODELU ---
# --- NAČTENÍ DAT A MODELU ---
@st.cache_data
def load_data():
    df = pd.read_csv('nba_data_final.csv')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

def get_latest_stats(df, team_name):
    team_matches = df[(df['TEAM_NAME_HOME'] == team_name) | (df['TEAM_NAME_AWAY'] == team_name)]
    last_match = team_matches.sort_values('GAME_DATE').iloc[-1]
    
    # OPRAVA: Změněno z ROLLING_PTS na ROLL_PTS
    if last_match['TEAM_NAME_HOME'] == team_name:
        return {
            'ELO': last_match['ELO_HOME'],
            'ROLL_PTS': last_match['ROLL_PTS_HOME']
        }
    else:
        return {
            'ELO': last_match['ELO_AWAY'],
            'ROLL_PTS': last_match['ROLL_PTS_AWAY']
        }

# --- HLAVNÍ LOGIKA ---
try:
    df = load_data()
    
    # Musíme definovat features, aby model věděl, co do něj leze
    features = ['ROLL_PTS_HOME', 'ROLL_PTS_AWAY', 'ELO_HOME', 'ELO_AWAY']

    # Pokud používáš joblib, musíš načíst všechny tři modely, které tvoje UI vyžaduje
    # Předpokládám, že tvoje pipeline ukládá jen model_win. 
    # Pro jednoduchost teď necháme trénování v aplikaci zapnuté, dokud neupravíme pipeline na ukládání všech 3 modelů.
    
    @st.cache_resource
    def train_all_models(data):
        X = data[features]
        m_win = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, data['HOME_WIN'])
        m_h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, data['PTS_HOME'])
        m_a = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, data['PTS_AWAY'])
        return m_win, m_h, m_a

    model_win, model_pts_h, model_pts_a = train_all_models(df)

    # --- SIDEBAR VÝBĚR ---
    teams_list = sorted(df['TEAM_NAME_HOME'].unique())
    home_team = st.sidebar.selectbox("🏠 Domácí tým", teams_list)
    away_team = st.sidebar.selectbox("🚀 Hostující tým", teams_list, index=1)

    if home_team != away_team:
        stats_h = get_latest_stats(df, home_team)
        stats_a = get_latest_stats(df, away_team)

        # Příprava vstupu (názvy sloupců musí přesně sedět na 'features')
        input_df = pd.DataFrame([[
            stats_h['ROLL_PTS'], 
            stats_a['ROLL_PTS'], 
            stats_h['ELO'], 
            stats_a['ELO']
        ]], columns=features)

        # VÝPOČTY
        prob_home = model_win.predict_proba(input_df)[0][1]
        pred_h_pts = model_pts_h.predict(input_df)[0]
        pred_a_pts = model_pts_a.predict(input_df)[0]

        # Zobrazení (tady pokračuje tvůj kód s progress bary...)
        st.success(f"Analýza hotova pro: {home_team} vs {away_team}")
        
        # Malý test pro tebe:
        if "Detroit" in home_team or "Detroit" in away_team:
            st.warning("⚠️ Pozor, Detroit má letos brutální formu!")

except Exception as e:
    st.error(f"Chyba: {e}")
