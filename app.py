import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="NBA AI Predictor 2026", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏀 NBA Smart Predictor 2026")
st.markdown("Analýza zápasů založená na **Elo Ratingu** a **Rolling Averages** z aktuální sezóny.")

@st.cache_data
def load_data():
    df = pd.read_csv('nba_data_final.csv')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

def get_latest_stats(df, team_name):
    team_matches = df[(df['TEAM_NAME_HOME'] == team_name) | (df['TEAM_NAME_AWAY'] == team_name)]
    last_match = team_matches.sort_values('GAME_DATE').iloc[-1]
    if last_match['TEAM_NAME_HOME'] == team_name:
        return {'ELO': last_match['ELO_HOME'], 'ROLL_PTS': last_match['ROLL_PTS_HOME']}
    else:
        return {'ELO': last_match['ELO_AWAY'], 'ROLL_PTS': last_match['ROLL_PTS_AWAY']}

try:
    df = load_data()
    features = ['ROLL_PTS_HOME', 'ROLL_PTS_AWAY', 'ELO_HOME', 'ELO_AWAY']

    @st.cache_resource
    def train_all_models(data):
        X = data[features]
        m_win = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, data['HOME_WIN'])
        m_h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, data['PTS_HOME'])
        m_a = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, data['PTS_AWAY'])
        return m_win, m_h, m_a

    model_win, model_pts_h, model_pts_a = train_all_models(df)

    # --- SIDEBAR ---
    teams_list = sorted(df['TEAM_NAME_HOME'].unique())
    home_team = st.sidebar.selectbox("🏠 Domácí tým", teams_list)
    away_team = st.sidebar.selectbox("🚀 Hostující tým", teams_list, index=1)

    if home_team != away_team:
        stats_h = get_latest_stats(df, home_team)
        stats_a = get_latest_stats(df, away_team)

        input_df = pd.DataFrame([[
            stats_h['ROLL_PTS'], stats_a['ROLL_PTS'], stats_h['ELO'], stats_a['ELO']
        ]], columns=features)

        # PREDIKCE
        prob_home = model_win.predict_proba(input_df)[0][1]
        pred_h_pts = model_pts_h.predict(input_df)[0]
        pred_a_pts = model_pts_a.predict(input_df)[0]

        # --- ZOBRAZENÍ VÝSLEDKŮ ---
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🏠 {home_team}")
            st.write(f"Aktuální Elo: **{int(stats_h['ELO'])}**")
            st.progress(prob_home)
            st.write(f"Pravděpodobnost výhry: **{prob_home:.1%}**")

        with col2:
            st.subheader(f"🚀 {away_team}")
            st.write(f"Aktuální Elo: **{int(stats_a['ELO'])}**")
            st.progress(1.0 - prob_home)
            st.write(f"Pravděpodobnost výhry: **{(1-prob_home):.1%}**")

        st.divider()
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Předpokládané skóre", f"{pred_h_pts:.1f} : {pred_a_pts:.1f}")
        with m2:
            spread = pred_h_pts - pred_a_pts
            st.metric("Spread (Handicap)", f"{spread:+.1f}")
        with m3:
            total = pred_h_pts + pred_a_pts
            st.metric("Total (Počet bodů)", f"{total:.1f}")

        if "Detroit" in home_team or "Detroit" in away_team:
            st.warning("⚠️ Pozor, Detroit má letos brutální formu!")
            
        st.info(f"Data byla naposledy aktualizována: {df['GAME_DATE'].max().strftime('%d.%m.%Y')}")

except Exception as e:
    st.error(f"Chyba: {e}")
