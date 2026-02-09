import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Načtení dat
df = pd.read_csv('nba_data_final.csv')

# 2. Výběr "Features" (vlastností), podle kterých budeme předpovídat
# Vybereme ty, které jsme si připravili
features = ['ROLLING_PTS_HOME', 'ROLLING_PTS_AWAY', 'ELO_HOME', 'ELO_AWAY']
X = df[features] # Vstupní data
y = df['HOME_WIN'] # To, co chceme předpovědět (0 nebo 1)

# 3. Rozdělení na trénovací a testovací sadu (80% učení, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vytvoření a trénování modelu
print("Trénuji model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Testování úspěšnosti
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- VÝSLEDKY ---")
print(f"Přesnost modelu: {accuracy:.2%}")
print("\nDetailní report:")
print(classification_report(y_test, y_pred))

# 6. Ukázka: Důležitost parametrů (Co nejvíc ovlivňuje výhru?)
importances = pd.Series(model.feature_importances_, index=features)
print("\nCo je pro model nejdůležitější:")
print(importances.sort_values(ascending=False))