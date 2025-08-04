# ✅ ML-Projekt Checkliste (für Industrie, Lebenslauf)

## 2. Dokumentation (README.md)

- [ ] Datenquelle & Beschreibung der Spalten
- [ ] Feature Engineering (One-Hot-Encoding, Umbenennungen)
- [ ] Modell: z. B. RandomForestClassifier + Parametereinstellungen
- [ ] API:
  - [ ] FastAPI-Inputstruktur
  - [ ] Beispiel-Request
- [ ] Power BI Report:
  - [ ] Beschreibung & Screenshots

## 3. Automatisierung

- [x] Erstelle ein ETL-Skript (`etl_predict.py`):
  - [x] Lade Daten
  - [x] Berechne Vorhersagen
  - [x] Speichere in SQL
- [ ] Zeitgesteuert ausführen:
  - [ ] Windows Task Scheduler / cron(start sql, create correct environment)

## 4. Deployment

- [ ] Dockerfile für API
- [ ] API-Zugriff absichern (Token, Auth)

## 5. Weiterentwicklung

- [x] Trainiere alternative Modelle (LogReg, XGBoost, etc.)
- [ ] Hyperparameter-Tuning mit GridSearch oder Optuna
- [ ] Feature Importance mit SHAP anzeigen
