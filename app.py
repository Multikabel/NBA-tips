import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="NBA AI Predictor 2026", layout="wide", page_icon="🏀")

# CSS pro hezčí vzhled
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏀 NBA Smart Predictor 2026")
st.markdown("Analýza zápasů založená na **Elo Ratingu** a **Rolling Averages** z aktuální sezóny.")

# --- NAČTENÍ DAT ---
@st.cache_data
def load_data():
    # Načte soubor, který GitHub Action aktualizuje každé ráno
    df = pd.read_csv('nba_data_final.csv')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

try:
    df = load_data()
    
    # --- TRÉNOVÁNÍ MODELŮ ---
    # Používáme RandomForest pro všechno - je robustní
    @st.cache_resource
    def train_all_models(data):
        features = ['ROLLING_PTS_HOME', 'ROLLING_PTS_AWAY', 'ELO_HOME', 'ELO_AWAY']
        X = data[features]
        
        # 1. Kdo vyhraje?
        model_win = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, data['HOME_WIN'])
        # 2. Kolik dá domácí?
        model_pts_h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, data['PTS_HOME'])
        # 3. Kolik dá host?
        model_pts_a = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, data['PTS_AWAY'])
        
        return model_win, model_pts_h, model_pts_a, features

    model_win, model_pts_h, model_pts_a, features = train_all_models(df)

    # --- SIDEBAR / VÝBĚR TÝMŮ ---
    st.sidebar.header("Nastavení analýzy")
    
    # Seznam unikátních týmů seřazený abecedně
    teams_list = sorted(df['TEAM_NAME_HOME'].unique())
    
    home_team = st.sidebar.selectbox("Domácí tým (Home)", teams_list, index=0)
    away_team = st.sidebar.selectbox("Hostující tým (Away)", teams_list, index=1)

    if home_team == away_team:
        st.sidebar.error("Musíš vybrat dva různé týmy!")
    else:
        # --- PREDIKCE ---
        # Získáme nejaktuálnější statistiky pro oba týmy z posledních odehraných zápasů
        latest_home = df[df['TEAM_NAME_HOME'] == home_team].iloc[-1]
        latest_away = df[df['TEAM_NAME_AWAY'] == away_team].iloc[-1]
        
        # Příprava dat pro model
        input_df = pd.DataFrame([[
            latest_home['ROLLING_PTS_HOME'], 
            latest_away['ROLLING_PTS_AWAY'], 
            latest_home['ELO_HOME'], 
            latest_away['ELO_AWAY']
        ]], columns=features)

        # Výpočty
        prob_home = model_win.predict_proba(input_df)[0][1]
        pred_h_pts = model_pts_h.predict(input_df)[0]
        pred_a_pts = model_pts_a.predict(input_df)[0]
        
        # --- ZOBRAZENÍ VÝSLEDKŮ ---
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🏠 {home_team}")
            st.write(f"Aktuální Elo: **{int(latest_home['ELO_HOME'])}**")
            st.progress(prob_home)
            st.write(f"Pravděpodobnost výhry: **{prob_home:.1%}**")

        with col2:
            st.subheader(f"🚀 {away_team}")
            st.write(f"Aktuální Elo: **{int(latest_away['ELO_AWAY'])}**")
            st.progress(1.0 - prob_home)
            st.write(f"Pravděpodobnost výhry: **{(1-prob_home):.1%}**")

        st.divider()
        
        # Metriky pro sázení / detailní analýzu
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric("Předpokládané skóre", f"{pred_h_pts:.1f} : {pred_a_pts:.1f}")
        with m2:
            spread = pred_h_pts - pred_a_pts
            st.metric("Spread (Handicap)", f"{spread:+.1f}")
        with m3:
            total = pred_h_pts + pred_a_pts
            st.metric("Total (Počet bodů)", f"{total:.1f}")

        st.info(f"Data byla naposledy aktualizována: {df['GAME_DATE'].max().strftime('%d.%m.%Y')}")

except Exception as e:
    st.error(f"Nepodařilo se načíst data nebo natrénovat model. Chyba: {e}")
    st.info("Ujisti se, že soubor 'nba_data_final.csv' existuje v tvém GitHub repozitáři.")
